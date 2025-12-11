import os
import subprocess
import time
import requests
import shutil
from mcrcon import MCRcon
import amulet
import asyncio

class MinecraftWorldManager:
    def __init__(self, server_dir="minecraft_server"):
        self.server_dir = server_dir
        self.world_name = "world"
        self.world_path = os.path.join(server_dir, self.world_name)
        self.jar_path = os.path.join(server_dir, "server.jar")
        self.eula_path = os.path.join(server_dir, "eula.txt")
        self.props_path = os.path.join(server_dir, "server.properties")
        
        # RCON settings
        self.rcon_host = "localhost"
        self.rcon_port = 25575
        self.rcon_pass = "password123"

    def reset_world(self):
        """Deletes the existing 'world' folder to ensure a fresh generation."""
        if os.path.exists(self.world_path):
            print(f"Deleting old world at: {self.world_path}...")
            # We use a loop because sometimes Windows holds onto files briefly
            try:
                shutil.rmtree(self.world_path)
                print("Old world deleted.")
            except Exception as e:
                print(f"Warning: Could not delete world folder completely: {e}")

    def setup_server(self, seed=""):
        """Downloads server, accepts EULA, and configures seed."""
        if not os.path.exists(self.server_dir):
            os.makedirs(self.server_dir)

        # 1. Download Server JAR (1.21.1)
        url = "https://piston-data.mojang.com/v1/objects/59353fb40c36d304f2035d51e7d6e6baa98dc05c/server.jar"
        if not os.path.exists(self.jar_path):
            print("Downloading Server JAR...")
            response = requests.get(url)
            with open(self.jar_path, "wb") as f:
                f.write(response.content)

        # 2. Accept EULA
        with open(self.eula_path, "w") as f:
            f.write("eula=true")

        # 3. Configure Server (RCON + Seed)
        # We explicitly write the properties file to set the seed
        config = f"""
            enable-rcon=true
            rcon.password={self.rcon_pass}
            rcon.port={self.rcon_port}
            level-name={self.world_name}
            level-seed={seed}
        """
        with open(self.props_path, "w") as f:
            f.write(config)
        print(f"Server configured with Seed: '{seed}'")

    def generate_world(self, radius_in_blocks):
        """
        Starts the server and forces generation within a radius around 0,0.
        radius_in_blocks: The distance from 0,0 to generate (e.g., 100 blocks)
        """
        print("--- Step 1: Generating World ---")
        
        # Start Server
        cmd = ["java", "-Xmx2G", "-jar", "server.jar", "nogui"]
        process = subprocess.Popen(cmd, cwd=self.server_dir, stdin=subprocess.PIPE)
        
        print("Server starting... waiting 30s for boot...")
        time.sleep(30) 

        try:
            with MCRcon(self.rcon_host, self.rcon_pass, port=self.rcon_port) as mcr:
                print("Connected via RCON.")
                
                # Convert blocks to chunk coordinates (radius / 16)
                # We round up to ensure we cover the requested area
                chunk_radius = (radius_in_blocks // 16) + 1
                
                min_c = -chunk_radius
                max_c = chunk_radius
                
                # Define coordinates: ~ is relative (but forceload uses absolute)
                min_x, min_z = min_c * 16, min_c * 16
                max_x, max_z = max_c * 16, max_c * 16
                
                print(f"Forcing generation (Radius: {radius_in_blocks} blocks)")
                print(f"Chunk area: {min_c},{min_c} to {max_c},{max_c}")
                
                mcr.command(f"forceload add {min_x} {min_z} {max_x} {max_z}")
                
                # Wait heuristic: 0.1s per chunk is usually enough for modern CPUs
                total_chunks = (max_c - min_c) * (max_c - min_c)
                wait_time = max(15, total_chunks * 0.1) 
                
                print(f"Waiting {int(wait_time)} seconds for generation...")
                time.sleep(wait_time)
                
                # Cleanup commands
                mcr.command("forceload remove all")
                mcr.command("save-all")
                time.sleep(2) # Give it a moment to write to disk
                mcr.command("stop")
        except Exception as e:
            print(f"RCON Error: {e}")
            process.terminate()

        process.wait()
        print("Server stopped.")

    async def query_block(self, x, y, z):
        """
        Asynchronously query a specific block.
        Runs in a separate thread to avoid blocking the main loop.
        """
        def _blocking_query():
            try:
                level = amulet.load_level(self.world_path)
                block = level.get_block(x, y, z, "minecraft:overworld")
                level.close()
                return block.namespaced_name
            except Exception as e:
                return f"Error: {e}"

        # Offload the blocking IO to a thread
        return await asyncio.to_thread(_blocking_query)

    async def find_blocks(self, block_name, search_radius_chunks=2):
        """
        Asynchronously scans for blocks. 
        Critical for scanning large areas without freezing your application.
        """
        print(f"--- Querying for '{block_name}' (Async) ---")

        def _blocking_search():
            try:
                level = amulet.load_level(self.world_path)
            except Exception as e:
                print(f"Error loading world: {e}")
                return []

            found_locations = []
            
            # Iterate chunks
            for cx in range(-search_radius_chunks, search_radius_chunks + 1):
                for cz in range(-search_radius_chunks, search_radius_chunks + 1):
                    try:
                        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
                        # Iterate blocks
                        for x in range(16):
                            for z in range(16):
                                # Full vertical scan (-64 to 320 for 1.21)
                                for y in range(-64, 320):
                                    try:
                                        block = chunk.get_block(x, y, z)
                                        if block_name in block.namespaced_name:
                                            global_x = (cx * 16) + x
                                            global_z = (cz * 16) + z
                                            found_locations.append((global_x, y, global_z))
                                    except:
                                        pass
                    except:
                        pass # Chunk not generated

            level.close()
            return found_locations

        # Offload the heavy search loop to a thread
        return await asyncio.to_thread(_blocking_search)

# --- EXECUTION ---
if __name__ == "__main__":
    manager = MinecraftWorldManager()
    
    # 1. Synchronous Setup (Server generation is still blocking/sync)
    # We generally don't make the server generation async because the OS process blocks anyway
    manager.reset_world()
    manager.setup_server(seed="12345")
    manager.generate_world(radius_in_blocks=80)
    
    # 2. Asynchronous Querying
    async def main():
        print("\n--- Starting Async Operations ---")
        
        # Example: Run a query and a search at the same time?
        # Or just await them one by one:
        
        # Query single block
        block_name = await manager.query_block(0, 70, 0)
        print(f"Block at 0,70,0: {block_name}")
        
        # Search for ores
        print("Starting ore search...")
        diamonds = await manager.find_blocks("diamond_ore", search_radius_chunks=5)
        
        print(f"Found {len(diamonds)} diamond ore blocks.")
        if diamonds:
            print(f"First location: {diamonds[0]}")

    # Run the async loop
    asyncio.run(main())