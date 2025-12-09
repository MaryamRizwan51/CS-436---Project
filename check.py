import os

# Check what files exist
data_dir = 'data2'
files = sorted([f for f in os.listdir(data_dir) if f.startswith('photo')])
print(f"You have {len(files)} photos: {files[0]} to {files[-1]}")

# Check if you already have a complete reconstruction
if os.path.exists('point_cloud.ply'):
    print("\n✅ You already have point_cloud.ply from Week 3")
    print("   → You might not need separate walls!")
else:
    print("\n❌ No point_cloud.ply found")
    print("   → You need to run Week 3 notebook OR create separate reconstructions")