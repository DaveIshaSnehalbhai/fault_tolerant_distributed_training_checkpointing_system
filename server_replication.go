package ikkat

import (
	"bufio"
	"context"
	pb "distributed-system-ikkat/filesystem"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"google.golang.org/grpc"                      // normal connecting
	"google.golang.org/grpc/codes"                // error codes
	"google.golang.org/grpc/credentials/insecure" // no credentials for security rn
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

// File Meta is same for all server
// primeSet is local cache of each server

const (
	HeartBeatTime        = 2 * time.Second // primary sends a HeartBeat message at every this interval
	PrimaryFailedTimeout = 5 * time.Second // if backups receive no heartbeat from primary for this long, primary is assumed to have failed
)

type Replication struct {
	Filename string
	Version  int32
	Content  []byte // for recovery, data of file
}

// Used for communication, server election
type ServerInfo struct {
	ID        string
	Address   string //localhost::8080
	Timestamp int64
	Alive     bool
}

// Used for recovery, consistency
type PersistentState struct {
	log []LogEntry
}

type LogEntry struct {
	Index    int    //Position in log
	Op       string // "WRITE", "DELETE"
	Filename string
	Content  []byte //This is data for operation
	Version  int32
}

type Heartbeat struct {
	PrimaryID string
	Timestamp int64
}

type Role int

const (
	Primary Role = iota
	Backup
)

// UpdateMessage is used for recovery and synchronization of out-of-date replicas, while normal replication is handled using log-based AppendEntries with majority acknowledgment
type UpdateMessage struct {
	IsFullSync bool
	LogEntries []LogEntry    // incremental sync
	FullFiles  []Replication // full snapshot
}

// revival works only if: recovered server receives heartbeat from current leader

// send a message with primaryID and timestamp
func (s *server) SendHeartbeat(ctx context.Context, req *pb.Heartbeat) (*pb.HeartbeatResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	srv, exists := s.servers[req.PrimaryId]
	if !exists {
		s.servers[req.PrimaryId] = ServerInfo{
			ID:        req.PrimaryId,
			Timestamp: req.Timestamp,
			Alive:     true,
		}
	} else {
		srv.Alive = true
		s.servers[req.PrimaryId] = srv
	}
	//Correct leader comparison
	if s.primaryID != "" {
		currentLeader := s.primaryID
		incomingLeader := req.PrimaryId

		if incomingLeader > currentLeader {
			// reject weaker leader
			return &pb.HeartbeatResponse{Success: false}, nil
		}
	}
	s.primaryID = req.PrimaryId
	s.lastHeartbeat = time.Now()
	return &pb.HeartbeatResponse{Success: true}, nil
}

// Leader -> Follower
func (s *server) sendHeartbeat(peer ServerInfo) {
	conn, _ := grpc.NewClient(peer.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	defer conn.Close()
	client := pb.NewHeartbeatServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	resp, err := client.SendHeartbeat(ctx, &pb.Heartbeat{
		PrimaryId: s.id,
		Timestamp: time.Now().Unix(),
	})
	if err != nil {
		return
	}
	if !resp.Success { // if rejected by followers, leader must step down
		fmt.Println("Another leader exists, stepping down")

		s.mu.Lock()
		s.role = Backup
		s.mu.Unlock()
	}
}

// for primary to send a heartbeat message to all backups every 2 seconds
func (s *server) StartHeartbeat() {
	for { // for infinity
		time.Sleep(HeartBeatTime)
		s.mu.Lock()
		isLeader := s.id == s.primaryID // if this is the leader
		s.mu.Unlock()
		if !isLeader {
			continue
		}
		for _, srv := range s.servers {
			if srv.ID == s.id { // don't send it to yourself
				continue
			} // send a heartbeat message to the rest
			s.sendHeartbeat(srv) // no goroutine explosion?
		}
	}
}

// If Hearbeat stops
func (s *server) MonitorPrimary() {
	for {
		time.Sleep(3 * time.Second)
		s.mu.Lock()
		if s.id == s.primaryID {
			s.mu.Unlock()
			continue
		}
		if time.Since(s.lastHeartbeat) > PrimaryFailedTimeout {
			fmt.Println("Primary failed, electing new leader")
			oldLeader := s.primaryID
			if srv, ok := s.servers[oldLeader]; ok {
				srv.Alive = false
				s.servers[oldLeader] = srv
			}
			newLeader := electPrimary(s.servers)
			s.primaryID = newLeader
			if s.id == newLeader {
				s.role = Primary
				fmt.Println("I am new primary")
				// when this backup becomes the new primary after a
				// failover, it must rebuild its filesPrimes from disk so
				// deduplication continues correctly. The old primary's in-memory
				// primeSet is gone; we reconstruct it from the committed files.
				go s.rebuildAllPrimeSets()
			} else {
				s.role = Backup
			}
		}
		s.mu.Unlock()
	}
}

// Backup keep track of var lastHeartbeat time.Time and also notes primary
func (s *server) OnHeartbeat(hb Heartbeat) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.primaryID = hb.PrimaryID
	s.lastHeartbeat = time.Now()
	if s.primaryID == "" || hb.PrimaryID < s.primaryID {
		s.primaryID = hb.PrimaryID
	}
	if s.id == s.primaryID { // if you are the primary
		s.role = Primary
	} else { // if you are not the primary update your role, jic, required if two think they are primary
		s.role = Backup
	}
	if srv, ok := s.servers[hb.PrimaryID]; ok {
		srv.Alive = true
		s.servers[hb.PrimaryID] = srv
	}
}

// Elect Primary
func electPrimary(servers map[string]ServerInfo) string {
	leader := ""
	for _, s := range servers {
		if !s.Alive {
			continue
		}
		if leader == "" || s.ID < leader {
			leader = s.ID
		}
	}
	return leader
}

// appending json logs in log file
// {"Index":1,"Op":"WRITE","Filename":"file.txt","Content":"...","Version":1}
func (s *server) appendToDisk(entry LogEntry) error {
	f, err := os.OpenFile(s.logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	_, err = f.Write(append(data, '\n'))
	if err != nil {
		return err
	}
	return f.Sync() // ensures durability, flushes buffered file data to disk immediately
}

// Recover from logs
// recoverFromLog replayed log entries by calling apply(), which
// correctly wrote files to disk. But it never called rebuildAllPrimeSets(),
// so after a crash+restart the primary's in-memory prime set was empty — it
// would accept already-seen primes again, breaking deduplication.

func (s *server) recoverFromLog() error {
	f, err := os.Open(s.logFilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // no log yet — fresh start
		}
		return err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	// default scanner buffer is 64 KB; log entries for large files
	// can easily exceed that, causing a "token too long" error that silently
	// stops recovery mid-log. Increase buffer to 16 MB.
	const maxBuf = 16 * 1024 * 1024
	buf := make([]byte, maxBuf)
	scanner.Buffer(buf, maxBuf)
	for scanner.Scan() {
		var entry LogEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			log.Println("recoverFromLog: skipping corrupt entry:", err)
			continue
		}
		s.log = append(s.log, entry)
		s.apply(entry)
		if entry.Index > s.commitIndex {
			s.commitIndex = entry.Index
			s.lastApplied = entry.Index
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	// Rebuild prime sets after log replay so deduplication resumes correctly
	s.rebuildAllPrimeSets()
	log.Printf("Recovery complete. commitIndex=%d entries=%d", s.commitIndex, len(s.log))
	return nil
}

// RebuildPrimeSet reconstructs the derived primeSet from file contents after recovery or replication.
func (s *server) RebuildPrimeSet(meta *FileMeta) error {
	if s.role != Primary { // only needed for primary
		return nil
	}
	filePath := filepath.Join(s.rootDir, meta.filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	primes := parseNumbers(data)
	newSet := make(FilePrimeSet)
	for _, p := range primes {
		newSet[p] = struct{}{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.filesPrimes[meta.filename] = &newSet
	return nil
}

func (s *server) getOrCreateFileMeta(filename string) *FileMeta {
	if entry, ok := s.table[filename]; ok {
		if meta, exists := s.files[entry.FD]; exists {
			return meta
		}
	}
	fd := s.nextFD
	s.nextFD++
	fullPath := filepath.Join(s.rootDir, filename)
	file, _ := os.OpenFile(fullPath, os.O_CREATE|os.O_RDWR, 0644)
	meta := &FileMeta{
		file:     file,
		filename: filename,
		version:  0,
		clients:  make(map[string]ClientState),
		// primeSet: make(map[uint64]bool),
	}
	s.files[fd] = meta
	s.table[filename] = &FileEntry{
		FD:      fd,
		version: 0,
	}
	return meta
}

// This is called when log entry is commited
// func (s *server) apply(entry LogEntry) {
// 	fm := s.getOrCreateFileMeta(entry.Filename)
// 	fm.mu.Lock()
// 	defer fm.mu.Unlock()
// 	path := filepath.Join(s.rootDir, entry.Filename)
// 	switch entry.Op {
// 	case "WRITE":
// 		err := os.WriteFile(path, entry.Content, 0644)
// 		if err != nil {
// 			log.Println("apply write failed:", err)
// 			return
// 		}
// 	case "DELETE":
// 		err := os.Remove(path)
// 		if err != nil && !os.IsNotExist(err) {
// 			log.Println("apply delete failed:", err)
// 			return
// 		}
// 	}
// 	fm.version = entry.Version // update version after success
// 	s.RebuildPrimeSet(fm)      // rebuild derived state
// }

func (s *server) apply(entry LogEntry) {
	switch entry.Op {

	case "WRITE":
		fullPath := filepath.Join(s.rootDir, entry.Filename)

		// Ensure directory exists
		err := os.MkdirAll(filepath.Dir(fullPath), 0755)
		if err != nil {
			log.Println("mkdir failed:", err)
			return
		}

		// ACTUAL FILE WRITE (this was missing)
		err = os.WriteFile(fullPath, entry.Content, 0644)
		if err != nil {
			log.Println("write failed:", err)
			return
		}

		log.Println("File written:", fullPath)

	case "TENSOR_WRITE", "TENSOR_META":
		// Removed: these op types were part of an alternate ML checkpoint
		// implementation (server_ml.go) that was never wired up as an RPC
		// and has been removed. ML checkpoint data now flows through the
		// normal "WRITE" path above via ServerHandle (server_handle.go),
		// which uses the existing Create/Open/Write/Close RPCs. This case
		// is kept only so that old log files containing these entries
		// (from before this cleanup) don't trigger "Unknown op" warnings
		// on replay; it intentionally does nothing.
		log.Println("apply: skipping legacy", entry.Op, "log entry for", entry.Filename)

	default:
		log.Println("Unknown op:", entry.Op)
	}
}

// Used when : Follower is behind OR restarted
// missing log , full state
// Recovery only
func (s *server) ApplyUpdate(msg *pb.UpdateMessage) error {
	s.mu.Lock()
	if msg.IsFullSync {
		s.log = nil
		s.commitIndex = 0
		s.lastApplied = 0
		for _, f := range msg.FullFiles {
			path := filepath.Join(s.rootDir, f.Filename)
			err := os.WriteFile(path, f.Content, 0644)
			if err != nil {
				s.mu.Unlock()
				return err
			}
			meta := s.getOrCreateFileMeta(f.Filename)
			meta.version = f.Version
			s.mu.Unlock() // unlock before expensive rebuild
			s.RebuildPrimeSet(meta)
			s.mu.Lock()
		}
		s.mu.Unlock()
		return nil
	}
	for _, e := range msg.LogEntries { // incremental logs
		entry := LogEntry{
			Index:    int(e.Index),
			Op:       e.Op,
			Filename: e.Filename,
			Content:  e.Content,
			Version:  e.Version,
		}
		if entry.Index <= len(s.log) {
			continue
		}
		if entry.Index != len(s.log)+1 {
			s.mu.Unlock()
			return fmt.Errorf("log gap")
		}
		s.log = append(s.log, entry)
		err := s.appendToDisk(entry)
		if err != nil {
			log.Println("persist failed:", err)
		}
	}
	s.mu.Unlock()
	return nil
}

// Called when leader sends a log entry
// Leader -> Follower: "store this operation"
// Follower: appends to log persists waits for commit
// AppendEntry -> append + persist ONLY
// Apply -> only after commit
// Recovery -> separate flow
func (s *server) AppendEntries(ctx context.Context, req *pb.AppendEntriesRequest) (*pb.AppendEntriesResponse, error) {

	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Append entries to log
	for _, e := range req.Entries {
		entry := LogEntry{
			Index:    int(e.Index),
			Op:       e.Op,
			Filename: e.Filename,
			Content:  e.Content,
			Version:  e.Version,
		}

		// Avoid duplicate append
		if entry.Index <= len(s.log) {
			continue
		}

		s.log = append(s.log, entry)

		//  persist log (this fixes missing log.txt)
		err := s.appendToDisk(entry)
		if err != nil {
			log.Println("backup log persist failed:", err)
		}
	}

	// 2. Update commit index
	if int(req.LeaderCommit) > s.commitIndex {
		s.commitIndex = int(req.LeaderCommit)
	}

	// 3. APPLY COMMITTED LOGS (THIS IS YOUR MISSING PIECE)
	s.applyCommitted()

	log.Println("Backup applied entries. commitIndex:", s.commitIndex)

	return &pb.AppendEntriesResponse{Success: true}, nil
}

// Only commited logs entry are appended.
func (s *server) applyCommitted() {
	for s.lastApplied < s.commitIndex && s.lastApplied < len(s.log) {

		s.lastApplied++
		// Get the log entry (index starts from 1)
		entry := s.log[s.lastApplied-1]
		// Apply to file system
		s.apply(entry)
	}
}

func sendAppendEntry(selfID string, peer ServerInfo, entry LogEntry, commitIndex int) bool {

	conn, err := grpc.NewClient(peer.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return false
	}
	defer conn.Close()
	client := pb.NewReplicationServiceClient(conn)
	req := &pb.AppendEntriesRequest{
		LeaderId:     selfID, // FIXED
		LeaderCommit: int32(commitIndex),
		Entries: []*pb.LogEntry{
			{
				Index:    int32(entry.Index),
				Op:       entry.Op,
				Filename: entry.Filename,
				Content:  entry.Content,
				Version:  entry.Version,
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 800*time.Millisecond)
	defer cancel()

	resp, err := client.AppendEntries(ctx, req)
	if err != nil {
		log.Println("AppendEntries RPC failed to", peer.Address, "error:", err)
		return false
	}

	if !resp.Success {
		log.Println("AppendEntries rejected by", peer.Address)
		return false
	}

	log.Println("Replicated to", peer.Address)
	return true
}

type WriteRequest struct {
	Filename string
	Data     []byte
}

func (s *server) HandleWrite(req WriteRequest) error {
	s.mu.Lock()
	meta := s.getOrCreateFileMeta(req.Filename)
	entry := LogEntry{
		Index:    len(s.log) + 1,
		Op:       "WRITE",
		Filename: req.Filename,
		Content:  req.Data,
		Version:  meta.version + 1,
	}
	s.log = append(s.log, entry)
	if err := s.appendToDisk(entry); err != nil {
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()

	// goroutines — data race identical to the one in Write/Close.
	// Fixed to use sync.Mutex-guarded counter (using sync.Mutex here
	// rather than atomic so the WaitGroup wait pattern is preserved).
	ackCount := 1
	commitIndex := entry.Index
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, peer := range s.servers {
		if peer.ID == s.id {
			continue
		}
		wg.Add(1)
		go func(p ServerInfo) {
			defer wg.Done()
			if sendAppendEntry(s.id, p, entry, commitIndex) {
				mu.Lock()
				ackCount++
				mu.Unlock()
			}
		}(peer)
	}
	wg.Wait()

	s.mu.Lock()
	alive := 0
	for _, sv := range s.servers {
		if sv.Alive {
			alive++
		}
	}
	s.mu.Unlock()

	majority := alive/2 + 1
	if ackCount >= majority {
		s.mu.Lock()
		s.commitIndex = entry.Index
		s.applyCommitted()
		s.mu.Unlock()
		return nil
	}
	return fmt.Errorf("failed to reach majority")
}

// Crash -> restart -> empty memory
// Follower -> RequestRecovery()
// Leader -> sends FullFiles
// Follower -> ApplyUpdate()
// After that Normal AppendEntries resumes
func (s *server) RequestRecovery(ctx context.Context, req *pb.RecoveryRequest) (*pb.UpdateMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.role != Primary {
		return nil, status.Errorf(codes.FailedPrecondition, "not leader")
	}
	var files []*pb.Replication
	for filename, entry := range s.table { // FIXED
		full := filepath.Join(s.rootDir, filename)
		data, err := os.ReadFile(full)
		if err != nil {
			continue
		}
		files = append(files, &pb.Replication{
			Filename: filename,
			Version:  entry.version,
			Content:  data,
		})
	}
	return &pb.UpdateMessage{
		IsFullSync: true,
		FullFiles:  files,
	}, nil
}

func (s *server) RecoverFromLeader() error {
	s.mu.Lock()
	leader, ok := s.servers[s.primaryID]
	log.Println("primary id", s.primaryID)
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("leader not found")
	}
	conn, err := grpc.NewClient(leader.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pb.NewRecoveryServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := client.RequestRecovery(ctx, &pb.RecoveryRequest{
		ServerId: s.id,
	})
	if err != nil {
		return err
	}
	return s.ApplyUpdate(resp)
}

// On server  startup
func (s *server) Start() {
	go func() {
		time.Sleep(2 * time.Second)
		s.RecoverFromLeader()
	}()
}

// Call this when server starts
func (s *server) Init() {
	s.lastHeartbeat = time.Now()
}

func (s *server) GetLeader(ctx context.Context, _ *emptypb.Empty) (*pb.LeaderResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Case 1: no leader yet
	if s.primaryID == "" {
		return nil, status.Errorf(codes.Unavailable, "no leader elected yet")
	}

	// Case 2: leader not found in map
	leader, ok := s.servers[s.primaryID]
	if !ok {
		return nil, status.Errorf(codes.Internal, "leader info missing")
	}

	return &pb.LeaderResponse{
		LeaderId: s.primaryID,
		Address:  leader.Address,
	}, nil
}
