Python dependencies of this repository are maintained with `hatch`.

The primary virtual environment is created in the `.venv` directory using

    hatch env create

To build the entire project execute

    hatch run build

Or from with the virtual environment created by `hatch`

    pytask build

Additional software, that must be present in the system includes:
- LaTeX (any current distribution)
- Inkscape with textext plugin installed

Artifacts generated during build are primarily placed in the `.build` directory, which is ignored by `git`, to not clutter the main directory tree.
The build directory is respected by LaTeX via the `latexmkrc` file.