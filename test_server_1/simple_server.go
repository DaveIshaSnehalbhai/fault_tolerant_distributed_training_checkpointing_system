package main

import (
	ikkat "distributed-system-ikkat"
	"flag"
	"log"
)

func main() {
	servers := map[string]ikkat.ServerInfo{
		"1": {ID: "1", Address: "localhost:5001", Alive: true},
		"2": {ID: "2", Address: "localhost:5002", Alive: true},
		"3": {ID: "3", Address: "localhost:5003", Alive: true},
	}
	id := flag.String("id", "1", "unique server id") // can get it from cli
	port := flag.String("port", "5001", "port to listen on")
	flag.Parse()
	root_dir := "storage"
	// We pass the ID and Port here so the server knows who it is.
	// Internally, NewServer can load the cluster map from a file or hardcoded list.
	srv := ikkat.NewServer(*id, *port, root_dir, servers)
	log.Printf("Starting Server %s on port %s...", *id, *port)
	if err := ikkat.StartServer(srv, *port); err != nil { // Recovery, Election, gRPC registration happens inside this call
		log.Fatalf("Critical server failure: %v", err)
	}
}
