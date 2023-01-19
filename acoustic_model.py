"""
Acoustic model using Hidden Markov Models (HMM) for 
"""
import numpy as np
import scipy.io


A_DICT = {
         "mean": [-1.820163e+000, 2.948045e-001, -9.102286e-001, -5.090310e-001, -4.833414e-001, -2.789246e-001, -2.483530e-001, -1.946086e-002, -2.488011e-001, -1.734584e-001, -1.326621e-001, -6.957074e-002, -6.207554e-002],
         "variance": [2.533771e+000, 3.355462e-001, 2.346364e-001, 1.186939e-001, 9.580594e-002, 6.385045e-002, 5.294375e-002, 4.059501e-002, 5.438076e-002, 3.317861e-002, 2.332679e-002, 1.959105e-002, 1.796214e-002]
         }

N_DICT = {
         "mean": [-3.409288e+000, 6.246719e-001, -4.743754e-002, -2.178264e-001, -5.912049e-001, -3.309700e-001, -3.621980e-001, -1.818327e-001, -3.115039e-001, -1.627000e-001, -1.629415e-001, -1.313724e-001, -1.055320e-001],
         "variance": [1.853842e+000, 2.350074e-001, 2.670902e-001, 1.030515e-001, 8.692522e-002, 5.001630e-002, 8.236017e-002, 4.526421e-002, 3.839431e-002, 2.944827e-002, 2.747891e-002, 1.762550e-002, 2.038012e-002]
         }

O_DICT = {
         "mean": [-2.774132e+000, 8.278207e-001, -5.067130e-001, -7.707845e-001, -4.396908e-001, -3.018584e-001, -4.529741e-001, -1.182087e-001, -5.049667e-002, -7.423452e-002, -1.431837e-001, -1.037804e-001, -6.737377e-002],
         "variance": [2.095176e+000, 2.614865e-001, 1.857337e-001, 1.567194e-001, 6.444157e-002, 6.257857e-002, 4.931529e-002, 5.150662e-002, 3.487385e-002, 2.563643e-002, 2.315338e-002, 1.673202e-002, 1.605465e-002]
         }

EMPTY_DICT = {
              "mean": np.zeros(13),
              "variance": np.zeros(13)
             }

A_probabilities = [[0.000000e+000, 1.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000],
[0.000000e+000, 8.653237e-001, 1.346763e-001, 0.000000e+000, 0.000000e+000], 
[0.000000e+000, 0.000000e+000, 8.373368e-001, 1.626632e-001, 0.000000e+000], 
[0.000000e+000, 0.000000e+000, 0.000000e+000, 8.573097e-001, 1.426903e-001], 
[0.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000, 0.000000e+000]] 

def probability_density(o_t,):
  pass


def multiply_vectors(vec1,vec2,vec3):
  result = 0
  for x1,x2,x3 in zip(vec1,vec2,vec3):
    result += x1*x2*x3
  return result

def b_func(o_t,mu_j,C_j,n=13):
  inv_C_j = np.zeros(13)
  for i in range(13):
    if (-1e-12 < C_j[i]  and C_j[i] < 1e-12):
      continue
    inv_C_j[i] = 1/C_j[i]

  determinant = np.prod(C_j)
  exponent = -0.5 * (multiply_vectors((o_t-mu_j),inv_C_j, (o_t-mu_j)))
  b = (1/np.sqrt(np.power(2*np.pi,n))) * (1/np.sqrt(determinant)) * np.exp(exponent)
  
  return b

"""
T - max time
N - max state
A_vector - Markov chain probabilities
b_vector - computed b
"""
def alpha_func(T,N,A_vector,b_vector):
  alphas = np.zeros((T,N))
  for t in range(T):
    if t==0:
      for j in range(1,N-1):
        alphas[t][j] = A_vector[0][j]*b_vector[0][j]
    else:
      for j in range(1,N-1):
        for i in range(1,N-1):
          alphas[t][j] = alphas[t][j] + alphas[t-1][i] * A_vector[i][j]*b_vector[t][j]
  
  final_probability = 0

  for i in range(1,N-1):
    final_probability += alphas[T-1][i] * A_vector[i][N-1]
  
  return alphas, final_probability

def beta_func(T,N,A_vector,b_vector):
  betas = np.zeros((T,N))
  for t in range(T-1,-1,-1):
    if t==T-1:
      for i in range(1,N-1):
        betas[t][i] = A_vector[i][N-1]
      print(betas[t])
    else:
      for i in range(1,N-1):
        for j in range(1,N-1):
          betas[t][i] += betas[t+1][j]*A_vector[i][j]*b_vector[t+1][j]
  
  final_probability = 0

  for i in range(1,N-1):
    final_probability += betas[0][i] * A_vector[0][i] * b_vector[0][i]
  
  return betas, final_probability




if __name__ == "__main__":
  loaded_text = np.loadtxt("./HMM/test_1.txt")
  b_list = []
  for segment in loaded_text:
    row = [0]
    for fonem in [A_DICT,N_DICT,O_DICT]:
      row.append(b_func(segment,fonem["mean"],fonem["variance"]))
    row.append(0)
    b_list.append(row)
    
  betas,afp = alpha_func(33,5,A_probabilities,b_list)
  print(np.log(afp))
  
  betas,bfp = beta_func(33,5,A_probabilities,b_list)
  print(np.log(bfp))
  exit()
  import csv
  with open("./out_beta.csv", 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    for row in betas:
        csvwriter.writerow(row)