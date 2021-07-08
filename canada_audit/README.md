# Web application to audit Canada's 2019 federal election

This folder contains code to generate a [bokeh](https://docs.bokeh.org/en/latest/index.html)-based web application 
which allows users to interactively select Canadian ridings to audit using some of RiLACS's confidence sequences.

Here's a quick demo of the app in action:

![audit](https://ian.waudbysmith.com/audit_demo_quick.gif)

A [longer demo](https://ian.waudbysmith.com/audit_demo.mov) is also available for download.

## Installing dependencies

If you do not already have `rilacs` installed,

```zsh
pip install git+ssh://git@github.com/WannabeSmith/RiLACS.git
```

In addition to the RiLACS package, this app depends on `bokeh` and `geopandas`.

```zsh
pip install bokeh geopandas
```

## Run the app

To start the web application,

```zsh
bokeh serve --show map.py
```
