Vancouver - 5 branches - 5 Years
================================

============== ===================
checksum32     2,220,417,399      
date           2019-10-23T16:26:19
engine_version 3.8.0-git2e0d8e6795
============== ===================

num_sites = 2, num_levels = 1, num_rlzs = ?

Parameters
----------
=============================== ================================
calculation_mode                'event_based'                   
number_of_logic_tree_samples    1                               
maximum_distance                {'default': 200.0}              
investigation_time              10.0                            
ses_per_logic_tree_path         1                               
truncation_level                3.0                             
rupture_mesh_spacing            10.0                            
complex_fault_mesh_spacing      10.0                            
width_of_mfd_bin                0.5                             
area_source_discretization      10.0                            
ground_motion_correlation_model None                            
minimum_intensity               {'SA(0.3)': 0.1, 'default': 0.1}
random_seed                     24                              
master_seed                     0                               
ses_seed                        23                              
=============================== ================================

Input files
-----------
======================= ==============================================
Name                    File                                          
======================= ==============================================
gsim_logic_tree         `gmmLT_analytical.xml <gmmLT_analytical.xml>`_
job_ini                 `job.ini <job.ini>`_                          
site_model              `Vs30_Vancouver.xml <Vs30_Vancouver.xml>`_    
source_model_logic_tree `ssmLT_0.xml <ssmLT_0.xml>`_                  
======================= ==============================================

Number of ruptures per source group
-----------------------------------
====== ========= ============ ============
grp_id num_sites num_ruptures eff_ruptures
====== ========= ============ ============
0      NaN       1,971        0.0         
1      NaN       42,533       0.0         
2      NaN       54,576       0.0         
====== ========= ============ ============

Slowest sources
---------------
========= ====== ==== ============ ========= ========= ============
source_id grp_id code num_ruptures calc_time num_sites eff_ruptures
========= ====== ==== ============ ========= ========= ============
========= ====== ==== ============ ========= ========= ============

Computation times by source typology
------------------------------------
==== =========
code calc_time
==== =========
A    0.0      
C    0.0      
==== =========

Information about the tasks
---------------------------
================== ======= ====== ======= ======= =======
operation-duration mean    stddev min     max     outputs
SourceReader       0.98894 NaN    0.98894 0.98894 1      
================== ======= ====== ======= ======= =======

Data transfer
-------------
==== ==== ========
task sent received
==== ==== ========

Slowest operations
------------------
====================== ======== ========= ======
calc_44501             time_sec memory_mb counts
====================== ======== ========= ======
composite source model 1.01031  0.0       1     
total SourceReader     0.98894  0.0       1     
====================== ======== ========= ======