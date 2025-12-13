from WorldGen.world_manager import MinecraftWorldManager

mc = MinecraftWorldManager()

mc.reset_world()
mc.setup_server(seed="12345")
mc.generate_world(radius_in_blocks=500)

def get_blocks_matrix(mc):
    mc