"""
Batch process the '.mat' mesh file
write the '.json' model file for VE or FE Solvers
"""
import numpy as np
import os
import shutil
from scipy.io import loadmat
from func_gene_JsonFile_Batch_beam1 import generate_json_file

if __name__ == "__main__":
    # 读取'.mat'文件路径
    mesh_folder_path = "./mesh/"

    # 生成'.json'文件路径
    json_folder_path = "./file/"

    # 文件路径预处理
    json_path_vem = os.path.join(json_folder_path, 'vem/') # 新建文件夹
    if os.path.exists(json_path_vem): # 强制删除文件夹及其所有子目录和文件
        shutil.rmtree(json_path_vem)             
    os.makedirs(json_path_vem) # 重新创建VEM文件夹

    json_path_fem = os.path.join(json_folder_path, 'fem/') # 新建文件夹
    if os.path.exists(json_path_fem): # 强制删除文件夹及其所有子目录和文件
        shutil.rmtree(json_path_fem)
    os.makedirs(json_path_fem) # 重新创建FEM文件夹

    m_count = 0
    j_count = 0
    mesh_list = os.listdir(mesh_folder_path)
    
    print(f"\n Path of json: {json_folder_path}")

    for mesh_name in mesh_list:
        if mesh_name.endswith('.mat'):
            # 加载网格
            m_count += 1
            mesh_path = os.path.join(mesh_folder_path, mesh_name)
            mesh_data = loadmat(mesh_path)
            
            # 生成json文件
            jcc = generate_json_file(json_path_vem, json_path_fem, mesh_name, mesh_data)
            j_count += jcc

    # TERMINAL输出
    print('{} VEM mesh(s) are read in'.format(m_count))
    print('{} json files are successfully generated'.format(j_count))
            
