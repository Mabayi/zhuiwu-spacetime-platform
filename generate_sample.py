"""Generate clearly synthetic terraced point cloud for pipeline verification."""
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent/'data';root.mkdir(exist_ok=True)
x,y=np.meshgrid(np.arange(0,180,.4),np.arange(0,120,.4))
u=x+3*np.sin(y/14)
z=np.zeros_like(x)
for start in (30,75,120):z+=np.clip((u-start)/8,0,1)*12
z+=.025*np.sin(x)*np.cos(y)
dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
a=np.zeros(x.size,dtype=dt)
for name,values in [('x',x),('y',y),('z',z)]:a[name]=values.ravel()
a['red']=(110+z.ravel()*2).astype('u1');a['green']=135;a['blue']=120
p=root/'synthetic_benches.ply'
header=f'ply\nformat binary_little_endian 1.0\ncomment SYNTHETIC TERRACED SURFACE - NOT MINE DATA\nelement vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
with p.open('wb') as f:f.write(header.encode('ascii'));f.write(a.tobytes())
print(f'Synthetic sample: {p}, {len(a)} points')
