# COMP9417 - Tutoring 23T1

Sina's slides for COMP9417 23T1 tutorials. 

I made this repo so I don't have to keep updating Dropbox when I change the slides, and so you can get the latest slides by just doing:

`$ git pull`

This also allows you to fork and make your own changes to the slides.

# Making your own slides

Please fork the repo so you can make your own changes separate from mine.

## Dependencies

This is likely the most painful part of the process, but once it's working, you'll never need to touch it again.

This repo is written for a UNIX system (Linux and Mac), I'm not sure of any way to get it working on Windows other than a WSL setup. 

You'll need:

- A LaTeX compiler (I use TeXlive)
	- Beamer addon - usually comes with a full LaTeX install
- Pandoc
- Make

## Making changes

Making slides is simple. I've written the slides in Markdown format and compile them using pandoc. The slides are in beamer/LaTeX, so you'll be fine if you know LaTeX and are comfortable with your text editor. Have a look through some source files and hopefully it becomes clear. If you get confused about the syntax just message me :)

## Compiling the slides

To make your own slides, just change a source file and run:

`$ make`

yes, it's that easy.
