# ACG

##  Demo Creation Process & Reproduction Guide:

First, run `run_simulation.py` using the following command:

``` bash
python run_simulation.py --scene ./data/scenes/fluid_rigid_coupling1.json
```

This will produce a directory called `fluid_rigid_coupling1_output` and in each subdirectory of it there is a `.png` file.

We can then generate a `.mp4` file using `make_video.py` with the following command:

```bash
python make_video.py --input_dir ./fluid_rigid_coupling1_output --image_name raw_view.png --output_path video.mp4 --fps 30
```

This will produce a file called `video.mp4`.

We change the file name to `video_for_fluid_rigid_coupling1.mp4` in the `demo` directory.

We can also produce `.obj` files for rendering. To do this, enter the `data/scenes/fluid_rigid_coupling1.json` file and in "Configuration" change "exportPly" and "exportObj" to be true. Then run the first command again. After that, run `surface_reconstruction.py` using the following command:

```bash
python surface_reconstruction.py --input_dir ./fluid_rigid_coupling1_output
```

After that, make a scene using blender. Since this `scene.blend` file we configured is rather large (over 100MB) and cannot be pushed to GitHub, we only reserve it in our local directories.

Run `render.py` to produce a directory `output_frames`.

```bash
python render.py
```

Finally, run `make_video_with_blender.py`:

```bash
python make_video_with_blender.py --input_dir ./output_frames --output_path rendered_video_for_fluid_rigid_coupling1.mp4
```

This produces the output video `rendered_video_for_fluid_rigid_coupling1.mp4` in `demo`.

We can also run the second demo:

```bash
python run_simulation.py --scene ./data/scenes/fluid_rigid_coupling2.json
python make_video.py --input_dir ./fluid_rigid_coupling2_output --image_name raw_view.png --output_path video.mp4 --fps 30
```

We change the name of the video and get `video_for_fluid_rigid_coupling2.mp4` in `demo`.



## Declarations:

### Direct References:

Part of the `data/models` and `data/scenes` in this project uses the data from [jason-huang03/SPH_Project](https://github.com/jason-huang03/SPH_Project/tree/master).

The `run_simulation.py` and `config_builder` in this project references the implementation from [erizmr/SPH_Taichi](https://github.com/erizmr/SPH_Taichi/tree/master).

The `surface_reconstruction.py` in this project uses [splashsurf](https://github.com/InteractiveComputerGraphics/splashsurf).

The `make_video.py` and `boundary.py` in this project references the implementation from [jason-huang03/SPH_Project](https://github.com/jason-huang03/SPH_Project/tree/master).