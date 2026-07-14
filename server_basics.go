package ikkat

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "distributed-system-ikkat/filesystem"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const (
	MaxOpenFiles     = 1000
	RequestCacheTTL  = 5 * time.Minute
	LeaseTimeout     = 10 * time.Minute
	ServerStorageDir = "storage"
)

type ClientState struct {
	lastSeen time.Time
	mode     pb.FileMode
}

type FileMeta struct {
	mu       sync.Mutex
	file     *os.File
	filename string
	version  int32
	clients  map[string]ClientState
}

type FilePrimeSet map[uint64]struct{}

type FileEntry struct {
	mu             sync.Mutex
	FD             int32
	cond           *sync.Cond
	activeReaders  int
	activeWriter   bool
	waitingWriters int
	version        int32
}

type RequestEntry struct {
	response  interface{}
	timestamp time.Time
}

type FileKey struct {
	clientID string
	filename string
}

type server struct {
	pb.UnimplementedFileServiceServer
	pb.UnimplementedReplicationServiceServer
	pb.UnimplementedRecoveryServiceServer
	pb.UnimplementedHeartbeatServiceServer
	pb.UnimplementedSnapshotServiceServer
	mu            sync.Mutex
	id            string
	role          Role
	primaryID     string
	servers       map[string]ServerInfo
	files         map[int32]*FileMeta
	table         map[string]*FileEntry
	nextFD        int32
	requests      map[string]*RequestEntry
	openMap       map[FileKey]int32
	rootDir       string
	log           []LogEntry
	commitIndex   int
	lastApplied   int
	lastHeartbeat time.Time
	logFilePath   string
	filesPrimes   map[string]*FilePrimeSet

	// Chandy-Lamport snapshot state
	snapshot *SnapshotState
}

func (s *server) saveVersion(filename string, version int32) {
	metaPath := filepath.Join(s.rootDir, filename+".meta")
	data := []byte(fmt.Sprintf("%d", version))
	os.WriteFile(metaPath, data, 0644)
}

func (s *server) loadVersion(filename string) int32 {
	metaPath := filepath.Join(s.rootDir, filename+".meta")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return 1
	}
	v, err := strconv.Atoi(string(data))
	if err != nil {
		return 1
	}
	return int32(v)
}

func (s *server) rebuildVersionTable() {
	files, err := os.ReadDir(s.rootDir)
	if err != nil {
		log.Println("Error reading directory:", err)
		return
	}
	for _, f := range files {
		name := f.Name()
		fullPath := filepath.Join(s.rootDir, name)
		if strings.HasSuffix(name, ".tmp") {
			os.Remove(fullPath)
			log.Println("Removed stale temp file:", name)
			continue
		}
		if strings.HasSuffix(name, ".meta") {
			continue
		}
		version := s.loadVersion(name)
		entry := &FileEntry{version: version}
		entry.cond = sync.NewCond(&entry.mu)
		entry.activeReaders = 0
		entry.activeWriter = false
		s.table[name] = entry
		log.Println("Recovered file:", name, "version:", version)
	}
}

// rebuildAllPrimeSets is called once after recovery to restore dedup state.
func (s *server) rebuildAllPrimeSets() {
	if s.role != Primary {
		return
	}
	if s.filesPrimes == nil {
		s.filesPrimes = make(map[string]*FilePrimeSet)
	}
	files, err := os.ReadDir(s.rootDir)
	if err != nil {
		log.Println("rebuildAllPrimeSets: ReadDir error:", err)
		return
	}
	for _, f := range files {
		name := f.Name()
		if strings.HasSuffix(name, ".tmp") || strings.HasSuffix(name, ".meta") {
			continue
		}
		if !strings.HasPrefix(name, "output") {
			continue
		}
		fullPath := filepath.Join(s.rootDir, name)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}
		primes := parseNumbers(data)
		newSet := make(FilePrimeSet, len(primes))
		for _, p := range primes {
			newSet[p] = struct{}{}
		}
		s.filesPrimes[name] = &newSet
		log.Printf("rebuildAllPrimeSets: %s — %d primes loaded", name, len(newSet))
	}
}

func NewFileEntry() *FileEntry {
	fe := &FileEntry{}
	fe.cond = sync.NewCond(&fe.mu)
	return fe
}

func (fe *FileEntry) CanRead() bool {
	return !fe.activeWriter && fe.waitingWriters == 0
}

func (fe *FileEntry) CanWrite() bool {
	return !fe.activeWriter && fe.activeReaders == 0
}

func (fe *FileEntry) AcquireReadNoPriority() {
	fe.mu.Lock()
	for fe.activeWriter {
		fe.cond.Wait()
	}
	fe.activeReaders++
	fe.mu.Unlock()
}

func (fe *FileEntry) AcquireRead() {
	fe.mu.Lock()
	for !fe.CanRead() {
		fe.cond.Wait()
	}
	fe.activeReaders++
	fe.mu.Unlock()
}

func (fe *FileEntry) ReleaseRead() {
	fe.mu.Lock()
	if fe.activeReaders > 0 {
		fe.activeReaders--
	}
	fe.cond.Broadcast()
	fe.mu.Unlock()
}

func (fe *FileEntry) AcquireWrite() {
	fe.mu.Lock()
	fe.waitingWriters++
	for !fe.CanWrite() {
		fe.cond.Wait()
	}
	fe.waitingWriters--
	fe.activeWriter = true
	fe.mu.Unlock()
}

func (fe *FileEntry) ReleaseWrite() {
	fe.mu.Lock()
	fe.activeWriter = false
	fe.cond.Broadcast()
	fe.mu.Unlock()
}

func (s *server) cleanupRequests() {
	for {
		time.Sleep(5 * time.Minute)
		s.mu.Lock()
		for id, entry := range s.requests {
			if time.Since(entry.timestamp) > RequestCacheTTL {
				delete(s.requests, id)
			}
		}
		s.mu.Unlock()
	}
}

func (s *server) cleanupLeases() {
	for {
		time.Sleep(time.Minute)
		type expiredLease struct {
			clientID string
			filename string
			mode     pb.FileMode
		}
		var expired []expiredLease
		s.mu.Lock()
		for fd, meta := range s.files {
			meta.mu.Lock()
			for clientID, client := range meta.clients {
				if time.Since(client.lastSeen) > LeaseTimeout {
					expired = append(expired, expiredLease{
						clientID: clientID,
						filename: meta.filename,
						mode:     client.mode,
					})
					delete(meta.clients, clientID)
					key := FileKey{clientID: clientID, filename: meta.filename}
					delete(s.openMap, key)
					fmt.Println("Lease expired client:", clientID, "FD:", fd)
				}
			}
			meta.mu.Unlock()
		}
		s.mu.Unlock()

		for _, e := range expired {
			entry := s.getFileEntry(e.filename)
			if e.mode == pb.FileMode_READ {
				entry.ReleaseRead()
			} else if e.mode == pb.FileMode_WRITE || e.mode == pb.FileMode(2) {
				entry.ReleaseWrite()
			}
		}
	}
}

func (s *server) FilterUniquePrimes(primes []uint64, primeSet FilePrimeSet) []uint64 {
	var unique []uint64
	for _, p := range primes {
		if _, exists := primeSet[p]; !exists {
			primeSet[p] = struct{}{}
			unique = append(unique, p)
		}
	}
	return unique
}

func (s *server) getFileEntry(name string) *FileEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.table[name]
	if !ok {
		entry = &FileEntry{version: 1}
		entry.cond = sync.NewCond(&entry.mu)
		s.table[name] = entry
	}
	if entry.cond == nil {
		entry.cond = sync.NewCond(&entry.mu)
	}
	return entry
}

func (s *server) Create(ctx context.Context, req *pb.CreateRequest) (*pb.OpenResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}
	s.mu.Lock()
	if entry, ok := s.requests[req.RequestId]; ok &&
		time.Since(entry.timestamp) < RequestCacheTTL {
		resp := entry.response.(*pb.OpenResponse)
		s.mu.Unlock()
		return resp, nil
	}
	s.mu.Unlock()

	safe, err := sanitizePath(req.Filename)
	if err != nil {
		return nil, err
	}
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")

	if !(strings.HasPrefix(safe, "output/") || strings.HasPrefix(safe, "output\\")) {
		return nil, status.Errorf(codes.PermissionDenied, "only output/")
	}

	full := filepath.Join(s.rootDir, safe)
	entry := s.getFileEntry(safe)

	entry.AcquireWrite()
	defer entry.ReleaseWrite()

	if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
		return nil, err
	}

	file, err := os.OpenFile(full, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0666)
	if err != nil {
		if os.IsExist(err) {
			return nil, status.Errorf(codes.AlreadyExists, "file exists")
		}
		return nil, err
	}

	s.mu.Lock()
	if len(s.files) >= MaxOpenFiles {
		s.mu.Unlock()
		file.Close()
		os.Remove(full)
		return nil, status.Errorf(codes.ResourceExhausted, "too many open files")
	}

	fd := s.nextFD
	s.nextFD++
	clientID := req.ClientId

	entry.mu.Lock()
	entry.version = 0
	currentVersion := entry.version
	entry.mu.Unlock()

	meta := &FileMeta{
		file:     file,
		filename: safe,
		version:  currentVersion,
		clients:  make(map[string]ClientState),
	}
	meta.clients[clientID] = ClientState{lastSeen: time.Now(), mode: pb.FileMode(req.Mode)}
	s.files[fd] = meta
	s.openMap[FileKey{clientID: clientID, filename: safe}] = fd

	resp := &pb.OpenResponse{Fd: fd, Version: currentVersion, Message: "file created"}
	s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
	s.mu.Unlock()

	return resp, nil
}

func (s *server) Delete(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}
	s.mu.Lock()
	if entry, ok := s.requests[req.RequestId]; ok &&
		time.Since(entry.timestamp) < RequestCacheTTL {
		resp := entry.response.(*pb.DeleteResponse)
		s.mu.Unlock()
		return resp, nil
	}
	s.mu.Unlock()

	safe, err := sanitizePath(req.Filename)
	if err != nil {
		return nil, err
	}
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")

	if strings.HasPrefix(safe, "input/") {
		return nil, status.Errorf(codes.PermissionDenied, "cannot delete input files")
	}

	full := filepath.Join(s.rootDir, safe)
	info, err := os.Stat(full)
	if err != nil && os.IsNotExist(err) {
		resp := &pb.DeleteResponse{Message: "File is already deleted"}
		s.mu.Lock()
		s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
		s.mu.Unlock()
		return resp, nil
	}
	if err == nil && info.IsDir() {
		return nil, status.Errorf(codes.InvalidArgument, "cannot delete directory")
	}

	s.mu.Lock()
	entry, exists := s.table[safe]
	s.mu.Unlock()
	if !exists {
		resp := &pb.DeleteResponse{Message: "File is already deleted"}
		s.mu.Lock()
		s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
		s.mu.Unlock()
		return resp, nil
	}

	entry.mu.Lock()
	if entry.activeReaders > 0 || entry.activeWriter {
		entry.mu.Unlock()
		return nil, status.Errorf(codes.FailedPrecondition, "file is currently in use by another client")
	}
	entry.activeWriter = true
	entry.mu.Unlock()

	defer func() {
		entry.mu.Lock()
		entry.activeWriter = false
		entry.cond.Broadcast()
		entry.mu.Unlock()
	}()

	if err := os.Remove(full); err != nil && !os.IsNotExist(err) {
		return nil, status.Error(codes.Internal, err.Error())
	}

	s.mu.Lock()
	delete(s.table, safe)
	if s.filesPrimes != nil {
		delete(s.filesPrimes, safe)
	}
	resp := &pb.DeleteResponse{Message: "File is deleted"}
	s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
	s.mu.Unlock()

	return resp, nil
}

func (s *server) Open(ctx context.Context, req *pb.FileRequest) (*pb.OpenResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}
	s.mu.Lock()
	if entry, ok := s.requests[req.RequestId]; ok &&
		time.Since(entry.timestamp) < RequestCacheTTL {
		resp := entry.response.(*pb.OpenResponse)
		s.mu.Unlock()
		return resp, nil
	}
	s.mu.Unlock()

	safe, err := sanitizePath(req.Filename)
	if err != nil {
		return nil, err
	}
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")

	full := filepath.Join(s.rootDir, safe)
	entry := s.getFileEntry(safe)
	mode := pb.FileMode(req.Mode)

	log.Println("Opening file:", full)

	if strings.HasPrefix(safe, "input/") {
		if mode != pb.FileMode_READ {
			return nil, status.Errorf(codes.PermissionDenied, "input files are read-only")
		}
		entry.AcquireReadNoPriority()
		defer entry.ReleaseRead()

		file, err := os.OpenFile(full, os.O_RDONLY, 0666)
		if err != nil {
			log.Println("ERROR opening input:", full, err)
			return nil, status.Errorf(codes.NotFound, "input file not found")
		}

		s.mu.Lock()
		defer s.mu.Unlock()

		if len(s.files) >= MaxOpenFiles {
			file.Close()
			return nil, status.Errorf(codes.ResourceExhausted, "too many open files")
		}

		fd := s.nextFD
		s.nextFD++
		clientID := req.ClientId
		meta := &FileMeta{
			file:     file,
			filename: safe,
			version:  entry.version,
			clients:  make(map[string]ClientState),
		}
		meta.clients[clientID] = ClientState{lastSeen: time.Now(), mode: mode}
		s.files[fd] = meta
		s.openMap[FileKey{clientID: clientID, filename: safe}] = fd

		log.Println("OPEN STORE:", clientID, "||", safe)
		resp := &pb.OpenResponse{Fd: fd, Version: entry.version, Message: "opened input file"}
		s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
		return resp, nil
	}

	// output file
	flags := os.O_RDONLY
	if mode == pb.FileMode_WRITE {
		flags = os.O_RDWR
	}
	file, err := os.OpenFile(full, flags, 0666)
	if err != nil {
		log.Println("ERROR opening output:", full, err)
		return nil, status.Errorf(codes.NotFound, "output file not found")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.files) >= MaxOpenFiles {
		file.Close()
		return nil, status.Errorf(codes.ResourceExhausted, "too many open files")
	}

	fd := s.nextFD
	s.nextFD++
	clientID := req.ClientId
	meta := &FileMeta{
		file:     file,
		filename: safe,
		version:  entry.version,
		clients:  make(map[string]ClientState),
	}
	meta.clients[clientID] = ClientState{lastSeen: time.Now(), mode: mode}
	s.files[fd] = meta
	s.openMap[FileKey{clientID: clientID, filename: safe}] = fd

	log.Println("OPEN STORE:", clientID, "||", safe)
	resp := &pb.OpenResponse{Fd: fd, Version: entry.version, Message: "opened output file"}
	s.requests[req.RequestId] = &RequestEntry{response: resp, timestamp: time.Now()}
	return resp, nil
}

func getClientIDFromContext(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	ids := md["clientid"]
	if len(ids) == 0 {
		return ""
	}
	return ids[0]
}

func (s *server) Close(ctx context.Context, req *pb.CloseRequest) (*pb.CloseResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}

	var meta *FileMeta
	var filename string
	var entry *FileEntry
	var mode pb.FileMode
	var dirty bool
	var tempPrimeSet FilePrimeSet

	clientID := getClientIDFromContext(ctx)
	reqID := req.RequestId
	dirty = req.Dirty

	s.mu.Lock()
	if entryCache, ok := s.requests[reqID]; ok &&
		time.Since(entryCache.timestamp) < RequestCacheTTL {
		if resp, ok := entryCache.response.(*pb.CloseResponse); ok {
			s.mu.Unlock()
			return resp, nil
		}
	}
	s.mu.Unlock()

	s.mu.Lock()
	m, ok := s.files[req.Fd]
	if !ok {
		s.mu.Unlock()
		return nil, status.Errorf(codes.NotFound, "file not open")
	}
	meta = m
	filename = meta.filename
	safe, _ := sanitizePath(filename)
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")
	fn := filename
	client, ok := meta.clients[clientID]
	if !ok {
		entry := s.getFileEntry(safe)
		entry.mu.Lock()
		version := entry.version
		entry.mu.Unlock()
		resp := &pb.CloseResponse{Message: "already closed", Version: version}
		s.requests[reqID] = &RequestEntry{response: resp, timestamp: time.Now()}
		s.mu.Unlock()
		return resp, nil
	}
	mode = client.mode
	s.mu.Unlock()

	entry = s.getFileEntry(safe)

	if dirty {
		entry.AcquireWrite()
		defer entry.ReleaseWrite()

		if mode != pb.FileMode_WRITE {
			return nil, status.Errorf(codes.PermissionDenied, "not opened in write mode")
		}
		if req.Version != entry.version {
			return nil, status.Errorf(codes.Aborted, "conflict")
		}

		s.mu.Lock()
		if s.filesPrimes == nil {
			s.filesPrimes = make(map[string]*FilePrimeSet)
		}
		primeSetPtr, ok := s.filesPrimes[fn]
		if !ok || primeSetPtr == nil {
			tempPrimeSet = make(FilePrimeSet)
		} else {
			// make a copy so we don't mutate live state until commit
			tempPrimeSet = make(FilePrimeSet, len(*primeSetPtr))
			for k, v := range *primeSetPtr {
				tempPrimeSet[k] = v
			}
		}
		s.mu.Unlock()

		log.Println("Processing size:", len(req.Data))
		nums := parseNumbers(req.Data)
		unique := s.FilterUniquePrimes(nums, tempPrimeSet)

		// build the full content of the file from the prime set
		// (not just the new unique ones — the whole accumulated set)
		var buf bytes.Buffer
		for p := range tempPrimeSet {
			fmt.Fprintf(&buf, "%d\n", p)
		}
		_ = unique // unique primes are already added to tempPrimeSet by FilterUniquePrimes

		data_to_log := buf.Bytes()
		logEntry := LogEntry{
			Index:    len(s.log) + 1,
			Op:       "WRITE",
			Filename: safe,
			Content:  data_to_log,
			Version:  entry.version + 1,
		}
		s.mu.Lock()
		s.log = append(s.log, logEntry)
		err := s.appendToDisk(logEntry)
		s.mu.Unlock()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "log persist failed")
		}

		var ackCount atomic.Int32
		ackCount.Store(1) // self
		commitIndex := s.commitIndex
		ackCh := make(chan bool, len(s.servers)-1)

		for _, peer := range s.servers {
			if peer.ID == s.id {
				continue
			}
			go func(p ServerInfo) {
				ok := sendAppendEntry(s.id, p, logEntry, commitIndex)
				ackCh <- ok
			}(peer)
		}

		if os.Getenv("FAIL_DURING_REPLICATION") == "1" {
			fmt.Println("Simulating server crash during replication")
			os.Exit(1)
		}

		majority := len(s.servers)/2 + 1
		timeout := time.After(1 * time.Second)
	ForLoop:
		for int(ackCount.Load()) < majority {
			select {
			case ok := <-ackCh:
				if ok {
					ackCount.Add(1)
				}
			case <-timeout:
				log.Println("Replication timeout")
				break ForLoop
			}
		}
		if int(ackCount.Load()) < majority {
			return nil, status.Errorf(codes.Unavailable, "failed to reach majority")
		}

		if os.Getenv("FAIL_BEFORE_COMMIT_SERVER") == "1" {
			fmt.Println("Simulating server crash before write commit")
			os.Exit(1)
		}

		s.mu.Lock()
		s.commitIndex = logEntry.Index
		s.applyCommitted()
		s.mu.Unlock()

		for _, peer := range s.servers {
			if peer.ID == s.id {
				continue
			}
			go sendAppendEntry(s.id, peer, logEntry, s.commitIndex)
		}

		// update prime set after successful commit
		s.mu.Lock()
		if ptr, ok := s.filesPrimes[fn]; ok && ptr != nil {
			*ptr = tempPrimeSet
		} else {
			newSet := make(FilePrimeSet, len(tempPrimeSet))
			for k, v := range tempPrimeSet {
				newSet[k] = v
			}
			s.filesPrimes[fn] = &newSet
		}
		s.mu.Unlock()
	}

	if os.Getenv("FAIL_AFTER_COMMIT_BEFORE_RESPONSE") == "1" {
		fmt.Println("Simulating server crash after commit but before response")
		os.Exit(1)
	}

	var newVersion int32
	entry.mu.Lock()
	if dirty {
		entry.version++
		s.saveVersion(safe, entry.version)
	}
	newVersion = entry.version
	entry.mu.Unlock()

	log.Println("DEBUG meta:", meta)
	s.mu.Lock()
	if meta != nil && meta.clients != nil {
		meta.mu.Lock()
		delete(meta.clients, clientID)
		meta.mu.Unlock()
	}
	s.mu.Unlock()

	resp := &pb.CloseResponse{Message: "closed", Version: newVersion}
	s.mu.Lock()
	s.requests[reqID] = &RequestEntry{response: resp, timestamp: time.Now()}
	s.mu.Unlock()

	return resp, nil
}

func (s *server) cleanupTempFiles() {
	files, err := os.ReadDir(s.rootDir)
	if err != nil {
		return
	}
	now := time.Now()
	for _, file := range files {
		name := file.Name()
		if !strings.HasSuffix(name, ".tmp") {
			continue
		}
		fullPath := filepath.Join(s.rootDir, name)
		info, err := file.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) < 2*time.Minute {
			continue
		}
		original := strings.TrimSuffix(name, ".tmp")
		entry := s.getFileEntry(original)
		entry.mu.Lock()
		inUse := entry.activeReaders > 0 || entry.activeWriter
		entry.mu.Unlock()
		if inUse {
			continue
		}
		if err := os.Remove(fullPath); err != nil {
			fmt.Println("cleanup failed:", err)
		}
	}
}

func (s *server) startCleanupRoutine() {
	go func() {
		for {
			time.Sleep(1 * time.Minute)
			s.cleanupTempFiles()
		}
	}()
}

func (s *server) Read(req *pb.ReadRequest, stream pb.FileService_ReadServer) error {
	if s.role != Primary {
		if s.primaryID == "" {
			return status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return status.Errorf(codes.Internal, "leader info missing")
		}
		return status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}
	ctx := stream.Context()
	clientID := getClientIDFromContext(ctx)

	safe, err := sanitizePath(req.Filename)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid path")
	}
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")

	key := FileKey{clientID: clientID, filename: safe}
	log.Println("READ LOOKUP:", clientID, "||", safe)

	s.mu.Lock()
	fd, ok := s.openMap[key]
	if !ok {
		s.mu.Unlock()
		return status.Errorf(codes.PermissionDenied, "file not opened by client")
	}
	meta, ok := s.files[fd]
	s.mu.Unlock()

	if !ok {
		return status.Errorf(codes.Internal, "file metadata missing")
	}

	client, ok := meta.clients[clientID]
	if !ok {
		return status.Errorf(codes.PermissionDenied, "client not registered")
	}

	mode := client.mode
	if mode != pb.FileMode_READ && mode != pb.FileMode_WRITE && mode != pb.FileMode(2) {
		return status.Errorf(codes.PermissionDenied, "read not allowed")
	}

	meta.mu.Lock()
	defer meta.mu.Unlock()

	client, exists := meta.clients[clientID]
	if !exists {
		return status.Errorf(codes.PermissionDenied, "client not registered for this file")
	}

	if time.Since(client.lastSeen) > LeaseTimeout {
		s.mu.Lock()
		delete(s.openMap, key)
		s.mu.Unlock()
		delete(meta.clients, clientID)
		return status.Errorf(codes.PermissionDenied, "lease expired")
	}

	client.lastSeen = time.Now()
	meta.clients[clientID] = client

	file := meta.file
	_, err = file.Seek(0, 0)
	if err != nil {
		return status.Errorf(codes.Internal, "seek failed")
	}

	buf := make([]byte, ChunkSize)
	for {
		n, err := file.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if err := stream.Send(&pb.ReadResponse{Data: chunk}); err != nil {
				return status.Errorf(codes.Internal, "stream send failed")
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return status.Errorf(codes.Internal, "read failed")
		}
	}
	return nil
}

func (s *server) Write(ctx context.Context, req *pb.WriteRequest) (*pb.WriteResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}

	var meta *FileMeta
	var filename string
	var entry *FileEntry
	var tmpFile *os.File
	var mode pb.FileMode
	var dirty bool
	var tempPrimeSet FilePrimeSet
	var logEntry LogEntry

	clientID := getClientIDFromContext(ctx)
	reqID := req.RequestId
	dirty = req.Dirty
	fullData := req.Data

	s.mu.Lock()
	if entryCache, ok := s.requests[reqID]; ok &&
		time.Since(entryCache.timestamp) < RequestCacheTTL {
		resp := entryCache.response.(*pb.WriteResponse)
		s.mu.Unlock()
		return resp, nil
	}
	m, ok := s.files[req.Fd]
	if !ok {
		s.mu.Unlock()
		return nil, status.Errorf(codes.NotFound, "file not open")
	}
	meta = m
	filename = meta.filename
	client, ok := meta.clients[clientID]
	if !ok {
		e := s.getFileEntry(filename)
		e.mu.Lock()
		version := e.version
		e.mu.Unlock()
		resp := &pb.WriteResponse{Message: "already closed", Version: version}
		s.requests[reqID] = &RequestEntry{response: resp, timestamp: time.Now()}
		s.mu.Unlock()
		return resp, nil
	}
	mode = client.mode
	s.mu.Unlock()

	entry = s.getFileEntry(filename)

	if dirty {
		entry.AcquireWrite()
		defer entry.ReleaseWrite()

		if mode != pb.FileMode_WRITE {
			return nil, status.Errorf(codes.PermissionDenied, "not opened in write mode")
		}
		if req.Version != entry.version {
			return nil, status.Errorf(codes.Aborted, "conflict")
		}

		logEntry = LogEntry{
			Index:    len(s.log) + 1,
			Op:       "WRITE",
			Filename: filename,
			Content:  fullData,
			Version:  entry.version + 1,
		}
		s.mu.Lock()
		s.log = append(s.log, logEntry)
		err := s.appendToDisk(logEntry)
		s.mu.Unlock()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "log persist failed")
		}

		var ackCount atomic.Int32
		ackCount.Store(1)
		commitIndex := s.commitIndex
		ackCh := make(chan bool, len(s.servers)-1)

		for _, peer := range s.servers {
			if peer.ID == s.id {
				continue
			}
			go func(p ServerInfo) {
				ackCh <- sendAppendEntry(s.id, p, logEntry, commitIndex)
			}(peer)
		}

		// count alive servers for majority
		s.mu.Lock()
		alive := 0
		for _, sv := range s.servers {
			if sv.Alive {
				alive++
			}
		}
		s.mu.Unlock()
		majority := alive/2 + 1

		timeout := time.After(1 * time.Second)
	WriteForLoop:
		for int(ackCount.Load()) < majority {
			select {
			case ok := <-ackCh:
				if ok {
					ackCount.Add(1)
				}
			case <-timeout:
				log.Println("Write: replication timeout")
				break WriteForLoop
			}
		}
		if int(ackCount.Load()) < majority {
			return nil, status.Errorf(codes.Unavailable, "failed to reach majority")
		}

		s.mu.Lock()
		s.commitIndex = logEntry.Index
		s.applyCommitted()
		s.mu.Unlock()

		for _, peer := range s.servers {
			if peer.ID == s.id {
				continue
			}
			go sendAppendEntry(s.id, peer, logEntry, s.commitIndex)
		}

		full := filepath.Join(s.rootDir, filename)
		tmp := full + ".tmp"

		safe, _ := sanitizePath(filename)
		safe = filepath.ToSlash(safe)
		safe = strings.TrimSpace(safe)
		safe = strings.TrimPrefix(safe, "/")

		s.mu.Lock()
		if s.filesPrimes == nil {
			s.filesPrimes = make(map[string]*FilePrimeSet)
		}
		primeSetPtr, ok := s.filesPrimes[safe]
		if !ok || primeSetPtr == nil {
			tempPrimeSet = make(FilePrimeSet)
		} else {
			tempPrimeSet = make(FilePrimeSet, len(*primeSetPtr))
			for k, v := range *primeSetPtr {
				tempPrimeSet[k] = v
			}
		}
		s.mu.Unlock()

		nums := parseNumbers(fullData)
		unique := s.FilterUniquePrimes(nums, tempPrimeSet)

		f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "temp open failed")
		}
		tmpFile = f

		for _, p := range unique {
			line := fmt.Sprintf("%d\n", p)
			if _, err := tmpFile.WriteString(line); err != nil {
				tmpFile.Close()
				return nil, status.Errorf(codes.Internal, "write failed")
			}
		}
		tmpFile.Sync()
		tmpFile.Close()

		src, err := os.Open(tmp)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to open tmp file: %v", err)
		}
		dst, err := os.OpenFile(full, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			src.Close()
			return nil, status.Errorf(codes.Internal, "failed to open destination: %v", err)
		}
		_, err = io.Copy(dst, src)
		if err != nil {
			src.Close()
			dst.Close()
			return nil, status.Errorf(codes.Internal, "copy failed: %v", err)
		}
		dst.Sync()
		src.Close()
		dst.Close()
		os.Remove(tmp)

		s.mu.Lock()
		if ptr, ok := s.filesPrimes[safe]; ok && ptr != nil {
			*ptr = tempPrimeSet
		} else {
			newSet := make(FilePrimeSet, len(tempPrimeSet))
			for k, v := range tempPrimeSet {
				newSet[k] = v
			}
			s.filesPrimes[safe] = &newSet
		}
		s.mu.Unlock()
	}

	var newVersion int32
	entry.mu.Lock()
	if dirty {
		entry.version++
		s.saveVersion(filename, entry.version)
	}
	newVersion = entry.version
	entry.mu.Unlock()

	resp := &pb.WriteResponse{Message: "write successful", Version: newVersion}
	s.mu.Lock()
	s.requests[reqID] = &RequestEntry{response: resp, timestamp: time.Now()}
	s.mu.Unlock()

	return resp, nil
}

func (s *server) TestAuth(ctx context.Context, req *pb.TestAuthRequest) (*pb.TestAuthResponse, error) {
	if s.role != Primary {
		if s.primaryID == "" {
			return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
		}
		leader, ok := s.servers[s.primaryID]
		if !ok {
			return nil, status.Errorf(codes.Internal, "leader info missing")
		}
		return nil, status.Errorf(codes.FailedPrecondition, "not leader: %s", leader.Address)
	}
	safe, err := sanitizePath(req.Filename)
	safe = filepath.ToSlash(safe)
	safe = strings.TrimSpace(safe)
	safe = strings.TrimPrefix(safe, "/")
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid path")
	}

	s.mu.Lock()
	entry, ok := s.table[safe]
	s.mu.Unlock()

	if !ok {
		return nil, status.Error(codes.NotFound, "file not found")
	}
	entry.mu.Lock()
	version := entry.version
	entry.mu.Unlock()
	return &pb.TestAuthResponse{Version: version}, nil
}
