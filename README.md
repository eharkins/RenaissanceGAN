# Renaissance GAN: one GAN model for Music and Image Generation
2017/18 CC CS Senior Project on Generative Adversarial Networks (GANs) - Eli, Calvin, Aidan, Nikhil

<img src="examples/song.gif" width="200" height = "200">
<img src="examples/eyes.png" width="300" height = "300">

Run with ```python3 unigan.py --input [Input .mid file, hdf5 file, or directory] --output [Ouput directory to be created]```

Optional arguments:

```
--epochs [How many epochs to stop after, default 12000] 
--batch [Batch size, default 5, 50 recommended for images]
--save-every [How often (in epochs) to save the image, default 100, 5 recommended for images]
--plot-every [How often (in epochs) to plot the image, default 100]
--no-display (Don't generate live display, necessary if X server is not set up.)

```

By default, GANiel will display a live feed of his generation, and, in the case of music, play generated files live.

<img src="examples/gwbush.gif" width="500" height = "500">
