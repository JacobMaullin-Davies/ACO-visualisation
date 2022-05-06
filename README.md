Visualisation of ACO project readme
---------------------

 * Introduction

-Index (Home) page

This system runs on a Django framework localhost.
A user can select an area in the index page using the 'toggle for area/location points'
When the toggle is switched, points can be selected within the area.
From the drop-down

-Visualisation (Visualisation) page

Accessed through the top navigation, the visualisation page will display the previously selected
area and points.
A user can select the type of search algorithm from the dropdown.
Parameters for the selected algorithm can be altered as desired.
The 'Toggle shown ants' toggle switch will display multiple ant paths
When correct parameters are selected, the 'Run Visualisation' button will initialise the
route optimisation and visualisation programs.
Once the environment has loaded, the displayed paths will be shown on the Map.
The 'Reset' button will clear all current paths on the map.
The iteration fitness overtime is shown on the chart below the map.


 * Requirements

 Anaconda - https://www.anaconda.com/
 OSMnx - street network data acquisition https://osmnx.readthedocs.io/en/stable/
 OpenRouteService API Key - provided in this application with free trial.
 Therefore limited number of requests can be made per hour/day.

 * Installation

 Firstly, install anaconda to workspace.  Once downloaded, open the Anaconda Prompt console

 Use the terminal or an Anaconda Prompt for the following steps:

 1. To create an environment:

 conda create -n djangoenv python=3.8.12 anaconda  

 (djangoenv can be any name for the environment, use specified python version )

 2. Proceed with 'y' when prompted

 proceed ([y]/n)?

 When the environment has been initialised, proceed to the next steps:

 3. Activate the environment

 conda activate djangoenv

 4. Installation of OSMnx

 conda install -c conda-forge osmnx

 5. Installation of Django

 conda install -c anaconda django

 (django should be version 3.1.2)


 * Configuration

 1. To run the django application, navigate to:

 cd ../env_django/aco_prototype

 2. In this directory, to run the server:

 python manage.py runserver

 (If you are on a Mac or Linux, use python3 manage.py runserver instead of the command given above)

 The terminal should output a similar message, with a URL will generated on the command line.

  System check identified no issues (0 silenced).
  May 06, 2022 - 15:51:30
  Django version 3.1.2, using settings 'Vis_aco.settings'
  Starting development server at http://127.0.0.1:8000/
  Quit the server with CTRL-BREAK.

  Click or copy this URL to a browser

  (Developed using Chrome Version 101.0.4951.54 (Official Build) (64-bit).
  System was not tested in other browsers)

 * Troubleshooting

    Ensure that OpenRouteService API key is valid:
    Navigate to directory:  ../env_django/aco_prototype/main

    in views.py file line 28

          ors_key = " "

    Ensure the key is valid, otherwise use own key

    OSMnx (imported as ox) might be slow due to API request of areas
    Area is not guaranteed to have available data (that is kept up to date)
    and request will not be prioritised due to free API, therefore loading times might be slow

    Developed using AMD Ryzen 7 3700X 8-Core Processor, 16.0 GB, Radeon RX 580 Series 8GB
    Windows 10
    Performance may be affected using a slower device system.
