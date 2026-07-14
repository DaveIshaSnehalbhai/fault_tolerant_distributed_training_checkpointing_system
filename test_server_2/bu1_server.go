package main

import (
	ikkat "distributed-system-ikkat"
	"flag"
	"log"
)

func main() {
	// 1. Get identity from CLI
	id := flag.String("id", "2", "unique server id")
	port := flag.String("port", "5002", "port to listen on")
	servers := map[string]ikkat.ServerInfo{
		"1": {ID: "1", Address: "localhost:5001", Alive: true},
		"2": {ID: "2", Address: "localhost:5002", Alive: true},
		"3": {ID: "3", Address: "localhost:5003", Alive: true},
	}
	flag.Parse()
	root_dir := "storage"
	// 2. Create the server object
	// We pass the ID and Port here so the server knows who it is.
	// Internally, NewServer can load the cluster map from a file or hardcoded list.
	srv := ikkat.NewServer(*id, *port, root_dir, servers)

	// 3. Just call StartServer
	// All the "magic" (Recovery, Election, gRPC registration) happens inside this call.
	log.Printf("Starting Server %s on port %s...", *id, *port)
	if err := ikkat.StartServer(srv, *port); err != nil {
		log.Fatalf("Critical server failure: %v", err)
	}
}
