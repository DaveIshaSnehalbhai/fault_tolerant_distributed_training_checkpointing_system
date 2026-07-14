package ikkat

import (
	"context"
	pb "distributed-system-ikkat/filesystem"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// NewServer is the ONLY way for an outsider to create a server object
func NewServer(id string, port string, rootDir string, servers map[string]ServerInfo) *server {
	return &server{
		id:          id,
		role:        Backup, // Default
		servers:     servers,
		table:       make(map[string]*FileEntry),
		requests:    make(map[string]*RequestEntry),
		rootDir:     rootDir,
		logFilePath: "./log_" + id + ".txt",
		files:       make(map[int32]*FileMeta),
		openMap:     make(map[FileKey]int32),
		filesPrimes: make(map[string]*FilePrimeSet),
	}
}

// function to start a server, handles the grpc initialisation portion
func StartServer(s *server, port string) error {
	address := ":" + port
	s.servers[s.id] = ServerInfo{
		ID:        s.id,
		Address:   "localhost:" + port,
		Timestamp: time.Now().Unix(),
		Alive:     true,
	}
	if err := s.recoverFromLog(); err != nil {
		log.Println("Recovery error:", err)
	}
	go func() {
		time.Sleep(2 * time.Second)
		if s.role == Backup {
			if err := s.RecoverFromLeader(); err != nil {
				log.Println("Recovery failed:", err)
			}
		}
	}()
	time.Sleep(1 * time.Second)           // allow cluster init
	s.primaryID = electPrimary(s.servers) // election
	if s.id == s.primaryID {
		s.role = Primary
	} else {
		s.role = Backup
	}
	log.Printf("[election] server %s: primary=%s role=%v", s.id, s.primaryID, s.role)
	go s.StartHeartbeat()
	go s.MonitorPrimary()
	inputDir := filepath.Join(s.rootDir, "input") // directories should exist
	outputDir := filepath.Join(s.rootDir, "output")
	os.MkdirAll(inputDir, os.ModePerm) // if already a dir, does nothing, returns nil
	os.MkdirAll(outputDir, os.ModePerm)
	lis, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	grpcServer := grpc.NewServer()
	pb.RegisterFileServiceServer(grpcServer, s)        // client api
	pb.RegisterReplicationServiceServer(grpcServer, s) // replication services
	pb.RegisterHeartbeatServiceServer(grpcServer, s)
	pb.RegisterRecoveryServiceServer(grpcServer, s)
	pb.RegisterSnapshotServiceServer(grpcServer, s)

	log.Println("Server started on", address)
	log.Println("Input dir:", inputDir)
	log.Println("Output dir:", outputDir)
	return grpcServer.Serve(lis)
}

// connecting client, hides grpc part from user
func DialClient(addresses []string) (*client, *grpc.ClientConn, error) {
	var lastErr error
	for _, addr := range addresses {
		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials())) // set up client connection, given address of server, no need of credentials
		if err != nil {
			lastErr = err
			continue
		}
		grpcClient := pb.NewFileServiceClient(conn) // FileServiceClient as defined using proto
		resp, err := grpcClient.GetLeader(context.Background(), &emptypb.Empty{})
		if err != nil {
			conn.Close()
			lastErr = err
			continue
		}
		leaderAddr := resp.Address
		if leaderAddr == addr {
			customClient := NewClient(grpcClient, conn)
			return customClient, conn, nil
		}
		conn.Close()
		leaderConn, err := grpc.NewClient(leaderAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			lastErr = err
			continue
		}
		leaderClient := pb.NewFileServiceClient(leaderConn)
		customClient := NewClient(leaderClient, leaderConn)
		return customClient, leaderConn, nil
	}
	return nil, nil, fmt.Errorf("failed to connect to leader: %v", lastErr)
}
