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
 OpenRouteService API Key - provided in this submission with free trial.
 Therefore limited number of requests can be made per hour/day.

 * Installation

 Firstly, install anaconda to workspace.  Once downloaded, open the Anaconda Prompt console

 Use the terminal or an Anaconda Prompt for the following steps:
 
 To create an environment: 

 conda create djangoenv 

 djangoenv can be any name for the environment

 Proceed with 'y' when prompted with proceed ([y]/n)?

 When the environment has been initialised, proceed to the next steps:

 conda activate djangoenv 
 
