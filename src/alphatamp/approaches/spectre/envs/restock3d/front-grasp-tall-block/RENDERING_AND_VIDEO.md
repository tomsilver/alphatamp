# Rendering & Video Guide (KinDER kinematic3D)

How to render frames and make videos of a KinDER kinematic3D scene (Shelf3D and
friends), at the exact level of detail used to produce this folder's
`demo_videos/*.mp4`. Written so another Claude Code session can reproduce it.

TL;DR: the scenes are PyBullet. The reliable way to render is to call
`pybullet_helpers.camera.capture_image(...)` directly against the env's PyBullet
client with a **custom camera** (the env's built-in `env.render()` uses a fixed
origin camera that frames the shelf poorly). Collect one frame per `env.step`
into a list, then write it with `imageio`.

---

## 0. Prerequisites (must be working before you can render a rollout)

Rendering a *rollout* runs the controllers, which use the Kinova IKFast solver.
That solver compiles a C++ module once on first use; if the build fails for
missing static LAPACK/BLAS, create a shim and point the linker at it:

```bash
mkdir -p /tmp/libshim
ln -sf /usr/lib/x86_64-linux-gnu/liblapack.so.3 /tmp/libshim/liblapack.a
ln -sf /usr/lib/x86_64-linux-gnu/libblas.so.3   /tmp/libshim/libblas.a
export LAPACK_DIR=/tmp/libshim BLAS_DIR=/tmp/libshim
# Python 3.12 also needs: pip install setuptools   (distutils shim)
```

Once compiled, the module is cached in the installed `pybullet_helpers` package
and no env vars are needed again. For **mp4** output: `pip install imageio-ffmpeg`
(GIF needs no extra deps). Run everything in the env where `kinder` /
`kinder_models` / `bilevel_planning` are installed.

> Rendering a *static* scene (no controller rollout) does NOT need IKFast — only
> stepping the arm controllers does.

---

## 1. The core primitive: `capture_image`

```python
from pybullet_helpers.camera import capture_image

frame = capture_image(
    physics_client_id,             # int: the PyBullet client to render
    camera_target=(1.0, 0.75, 0.35),  # 3D point the camera looks at (world coords)
    camera_distance=3.9,           # meters from target to camera
    camera_yaw=38,                 # azimuth degrees (rotation about world +z)
    camera_pitch=-31,              # elevation degrees (negative = looking down)
    image_width=640,
    image_height=480,
    # defaults you rarely change:
    near_val=0.1, far_val=100.0, fov=60.0,
)
# frame is an np.ndarray, dtype uint8, shape (image_height, image_width, 3), RGB.
```

Full signature (from `pybullet-helpers/src/pybullet_helpers/camera.py`):

```
capture_image(physics_client_id, camera_distance=1.5, camera_yaw=0,
              camera_pitch=-15, camera_target=(0, 0, 0.5),
              image_width=1674, image_height=900, near_val=0.1, far_val=100.0,
              fov=60.0, specify_position=False, camera_position=(0, 0, 0),
              camera_orientation=(0, 0, 0, 1)) -> np.ndarray[uint8]
```

There are two coordinate conventions available:
- **Orbit camera (used here):** set `camera_target`, `camera_distance`,
  `camera_yaw`, `camera_pitch`. The camera orbits the target. This is by far the
  easiest to aim.
- **Explicit pose:** set `specify_position=True`, `camera_position=(x,y,z)`,
  `camera_orientation=(x,y,z,w)`. Only needed for exact camera poses.

### Getting the PyBullet client id (and robot body ids)

The env wraps an inner object-centric env that owns the PyBullet client:

```python
oce = env.unwrapped._object_centric_env      # ObjectCentricShelf3DEnv
pcid   = oce.physics_client_id               # -> pass to capture_image
arm_id = oce.robot.arm.robot_id              # arm body id (for collision checks)
base_id = oce.robot.base.robot_id            # mobile-base body id
```

> IMPORTANT — there are two PyBullet clients in play. The **env** client
> (`oce.physics_client_id`) holds the *executed* state (what you want to film).
> The bilevel `models` build their own internal planning sim with a *separate*
> client. Always render from the **env** client to see the real rollout.

---

## 2. Choosing camera parameters (scene coordinate reference)

Coordinates are world frame, **z up**, meters. For Shelf3D-o1:

| Object | Where |
|---|---|
| Robot start | near the origin (base ~`(0,0)`) |
| Block spawn | on the floor, `x ∈ [-1, 1]`, `y ∈ [-1, 1]`, center z ≈ 0.06 |
| Shelf / cupboard | at `(2.0, 2.4)`; 4 boards at z ≈ 0.02 / 0.29 / 0.55 / 0.82 |

So the whole task spans roughly `x ∈ [-1, 2]`, `y ∈ [-1, 2.4]`, `z ∈ [0, 0.85]`.

How the four orbit knobs map to framing:
- `camera_target` = the point you center on. Use the **midpoint of the action**
  for a whole-task shot, or a specific object for a close-up.
- `camera_distance` = zoom (bigger = further back = more in frame).
- `camera_yaw` = which side you view from (spin around the target).
- `camera_pitch` = how steeply you look down (−90 = straight top-down, −10 =
  nearly level).

---

## 3. The whole-task camera used for `demo_videos/`

This frames both the floor pick area and the shelf in one fixed shot:

```python
def render(env):
    oce = env.unwrapped._object_centric_env
    return capture_image(
        oce.physics_client_id,
        camera_target=(1.0, 0.75, 0.35),   # midpoint between block area and shelf
        camera_distance=3.9,               # far enough to see both ends
        camera_pitch=-31,                  # tilt down
        camera_yaw=38,                     # 3/4 view
        image_width=640,
        image_height=480,
    )
```

Pick and place happen at opposite ends of the room, so a single fixed camera is
a compromise: the block/shelf look small but the whole motion is visible. For a
crisper shot of just one phase, use a close-up (Section 6) or render two
segments with different cameras and concatenate the frame lists.

---

## 4. The frame-capture loop

Capture one frame after `reset` and one after every `env.step`:

```python
frames = [render(env)]                     # initial frame
obs, info = env.reset(seed=123)
frames = [render(env)]                      # re-capture after reset
# ... drive the controllers / agent, appending a frame each step ...
for _ in range(max_steps):
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(render(env))
    if terminated or truncated:
        break
```

The canonical, complete example is **`demo_front_shelf3d.py`** in this folder:
`render()` (the camera) + `_run_controller()` / `demo_planner()` (the loops).
Copy its structure. It supports both a direct-controller rollout and a full
SeSamE-planner rollout.

---

## 5. Writing the video

`imageio` turns the frame list into a file. The extension picks the format.

### mp4 (small, recommended — needs `imageio-ffmpeg`)

```python
import imageio.v2 as iio
iio.mimsave("out.mp4", frames, fps=30, macro_block_size=16)
```

- `fps`: 30 looks smooth. The env advertises `env.metadata["render_fps"]` (= 20).
- `macro_block_size=16`: mp4 codecs want width/height divisible by 16. **640×480
  is safe** (640/16=40, 480/16=30). If your dims aren't divisible by 16, imageio
  silently resizes (or pass `macro_block_size=1` to disable, risking player
  incompatibility). Prefer choosing divisible dims up front.
- Typical size: ~220 KB for ~360 frames at 640×480. (A GIF of the same is ~50 MB
  — avoid GIF for anything long.)

### GIF (no extra deps, but large)

```python
import imageio.v2 as iio
iio.mimsave("out.gif", frames, fps=20, loop=0)   # loop=0 = loop forever
```

GIF **de-duplicates identical consecutive frames**, so a stalled/static segment
collapses — the saved frame count can be far less than the number of steps.
(That is itself a useful signal: if a GIF has far fewer frames than steps, the
motion froze.)

---

## 6. Diagnostic renders (close-ups, top-down, single frames)

For inspecting a specific moment (e.g., "is the gripper on the right faces?"),
render a tight custom camera at that instant. Get the object's position from the
state to target it:

```python
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
st = models.observation_to_state(obs)
c  = st.get_objects(Kinematic3DCuboidType)[0]
cx, cy, cz = (st.get(c, f) for f in ("pose_x", "pose_y", "pose_z"))
oce = env.unwrapped._object_centric_env
```

Cameras that were useful here:

```python
# Top-down onto the grasp (shows which faces the fingers contact):
capture_image(oce.physics_client_id, camera_target=(cx, cy, cz),
              camera_distance=0.4, camera_pitch=-89, camera_yaw=0,
              image_width=560, image_height=560)

# 3/4 close-up of the grasp:
capture_image(oce.physics_client_id, camera_target=(cx, cy, cz + 0.03),
              camera_distance=0.55, camera_pitch=-25, camera_yaw=45)

# Axis-aligned side views (to read grip geometry along an axis):
capture_image(..., camera_target=(cx, cy, cz), camera_distance=0.55,
              camera_yaw=0,  camera_pitch=-10)   # look along one axis
capture_image(..., camera_yaw=90, camera_pitch=-10)  # look along the other

# Close-up of the placed block on the shelf:
capture_image(oce.physics_client_id, camera_target=(2.0, 2.35, 0.4),
              camera_distance=1.15, camera_pitch=-8, camera_yaw=-25)
```

Save a single frame as PNG and **view it with the Read tool** (Claude Code
renders images):

```python
import imageio.v2 as iio
iio.imsave("/tmp/frame.png", capture_image(oce.physics_client_id, ...))
# then: Read /tmp/frame.png
```

---

## 7. Inspecting an existing mp4/gif (extract frames to look at)

```python
import imageio.v2 as iio
frames = iio.mimread("out.mp4", memtest=False)   # list of HxWx3 arrays
# or, v3: import imageio.v3 as iio3; frames = iio3.imread("out.mp4", index=None)
print(len(frames), frames[0].shape)
iio.imsave("/tmp/f150.png", frames[150])         # save one frame -> Read it
```

Reading a saved PNG back with the Read tool is how you visually check a render
inside a session.

---

## 8. Alternatives (for completeness)

- **`env.render()`** (built-in, rgb_array): returns a frame using the env
  config's fixed camera — for Shelf3D that's `camera_target=(0,0,0)`,
  `camera_yaw=0`, `camera_distance=2.0`, `camera_pitch=-20`, 640×360. Convenient
  but poorly framed for the shelf; the custom `capture_image` above is preferred.
- **`gymnasium.wrappers.RecordVideo`**: wrap the env
  (`RecordVideo(env, folder, episode_trigger=lambda _: True)`) and it writes an
  mp4 per episode via moviepy using `env.render()` (so, the fixed camera). This
  is what `kinder-bilevel-planning/experiments/run_experiment.py` uses with
  `make_videos=True`. Needs `moviepy`; less control over framing.

---

## 9. Gotchas / tips

- **Render from the env client, not the planning sim.** (Section 1.)
- **mp4 dimensions divisible by 16** (Section 5). 640×480 is the safe default.
- **IKFast build** must succeed before any rollout renders (Section 0).
- **Static scene, no rollout:** you can render a hand-set state — build the env,
  `env.reset(seed=...)`, optionally `oce.set_state(state)` (needs
  `allow_state_access=True` on `kinder.make(...)`), then `capture_image(...)`.
  No IKFast needed for a static render.
- **Frame rate vs. real time:** frames are one-per-`env.step`; the video's
  wall-clock duration is `len(frames)/fps`. Raise `fps` to speed up long
  rollouts, or subsample (`frames[::2]`) before `mimsave`.
- **Colors are RGB** already (no BGR swap needed) — `capture_image` returns RGB.
