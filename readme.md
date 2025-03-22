This repository contains all files necessary to generate the plots in the paper [Minimax Optimal Estimation of Stability Under Distribution Shift](https://arxiv.org/abs/2212.06338). 

## File Structure

- **Simulation Stability.ipynb**  
  <sub>All the code to generate results for the simulation part: Figures 2 and 3.</sub>

- **Stability_Statistical_Model_NHIS.ipynb**  
  <sub>All the code to generate results for the Health utilization prediction model (NHIS) in Section 5.2 and the appendix.</sub>

- **stability_queue_code/**  
  <sub>All the code to generate results for queueing systems in Section 5.1 and the appendix.</sub>
  - `model_train_3920_2022-01-28_04/01/39.pkl`  #the DQN model trained using the file `train_DQN.py`, which is also the DQN model in the paper
  - `Stability_queue_convergence.py` # generate data for Figure 5
  - `Stability_queue_functions.py` # all functions necessary for  queueing systems
  - `Stability_queue_kde.py` # generate data for Figure 5 (the KDE estimator)
  - `Stability_queue.ipynb` # code to generate plots for  queueing systems
  - `train_DQN.py` # the script to train the DQN model `model_train_3920_2022-01-28_04/01/39.pkl`
 
## Data

- **Simulation**
  
  Data is generated in the notebook.

- **NHIS**
  
  Data is available upon request.  
- **Queueing models**
  
  To run the queueing systems code for Figure 4 and Figures 7 - 11, one needs to generate data using the scripts given in  `Stability_queue.ipynb`.
  For example, in the below code block, the first line of code generates data and saved it to a file, the second line read the saved data and perform plotting
  Note that each file will contain a time of generation for versioning purposes: one should modify the time appropriately when reading the file. In the below example, one should change `_2023-08-14_12:59:34` appropriately.

  ```python
  evaluate_policy_under_distribution(arrival_distribution_t1, service_distribution_t1, mixture_prob_arrival, mixture_prob_service,'distribution_shift_type_1_new',number_of_simulations = 10000)
  data_0 = joblib.load('./Result/model_data/distribution_shift_type_1_10K_data__2023-08-14_12:59:34.pkl')
  evaluate_policy_under_distribution(arrival_distribution_t1, service_distribution_t1, mixture_prob_arrival, mixture_prob_service,'distribution_shift_type_1_new',1,data_0)
  ```
  
## Creating the Environment

To repliacte the environment for all code, you can install all the required pacakges using the following command, note that not all pacakges are necessary, since most packages in this project is standard, we recommend manually installing if any bugs occur from running the below code.

```bash
conda create -n stability_env python=3.10.12
conda activate stability_env
pip install -r requirements.txt
```


## Additional instructions

- All file paths in the code needs to be changed appropriately, for example, `file_path = '/user/ym2865/NHIS/model_file/'` in the code needs to be changed to your directory
- For each code in all notebooks, there is a description on what this code corresponds to (e.g., generate data for Figure 1, or generate Figure 1).

## Authors

* **[Hongseok Namkoong](https://hsnamkoong.github.io/)**, **[Yuanzhe Ma](https://yuanzhe-ma.com/)**, and **[Peter W. Glynn](https://web.stanford.edu/~glynn/)**
* For any questions, please contact Yuanzhe Ma at ym2865@columbia.edu.



