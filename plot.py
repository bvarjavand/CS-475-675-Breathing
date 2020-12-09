# plot (optional)
points = np.array([df['Time'], df['Volume']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, ax = plt.subplots(1,1)

# Use a boundary norm
cmap = ListedColormap(['k', 'r', 'g', 'b'])
norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(df['Label'])
lc.set_linewidth(2)
line = ax.add_collection(lc)

ax.set_xlim(df['Time'].min(), df['Time'].max())
ax.set_ylim(df['Volume'].min(), df['Volume'].max())
plt.savefig("example.png")
plt.show()