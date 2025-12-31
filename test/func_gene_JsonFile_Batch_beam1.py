"""
function of Generate_JsonFile_Batch.py
"""
import numpy as np
import math as mth
import os
import json
import random
from Generate_JsonFile import IdentiBound, convert_to_2d_array


"""def transformations():
    return [
        (lambda x, y: (x, y)),                    # 0度旋转
        (lambda x, y: (-y, x)),                   # 90度逆时针旋转
        (lambda x, y: (-x, -y)),                 # 180度旋转
        (lambda x, y: (y, -x)),                  # 270度旋转
        (lambda x, y: (x, -y)),                    # 沿x轴对称
        (lambda x, y: (-x, y))                     # 沿y轴对称
    ]"""

# 不考虑旋转
def transformations():
    return [
        (lambda x, y: (x, y)),                    # 0度旋转
    ]

def generate_json_file(vepath, fepath, mname, mdata):
    j_count = 0

    # 文件名预处理(修改后缀)
    ext = os.path.splitext(mname)
    jname = ext[0] + '.json'

    # 网格预处理
    ve_node = mdata['node']
    ve_elem = mdata['elem']
    ve_elem = convert_to_2d_array(ve_elem)

    # eq_d = mth.ceil(mth.sqrt(len(ve_elem))) # fem等效离散单元个数
    eq_d = 8                                 # 固定fem单元个数
    fe_node, fe_elem = generate_quad_mesh(eq_d)

    # 写入字典生成json
    material = [6.5e3, 0.0] # 材料属性, E=1MPa(1e6Pa)
    u_list = np.array([0.2]) # 给定若干(x-coor)位移值

    # 坐标变换[4旋转 + 2对称]
    xy_trans = transformations()

    for k in range(len(u_list)): # NO.1 - 对位移值循环
        # 生成本质边界条件
        value_u = u_list[k]      
        coor_b, disp_b = generate_bound_cond(value_u)

        # 生成面力边界条件
        '''新增定义面力在y=1.0处'''
        f_x = np.array([0.0])
        f_y = np.array([-1])
        if len(f_x) == len(f_y) == len(coor_b):
            n_bc_vem = generate_f_cond(ve_node, f_x, f_y)
            n_bc_fem = generate_f_cond(fe_node, f_x, f_y)
        else:
            print('!!!ERROR:面力工况数与本质边界条件数不一致')
            break
        
        # 生成json文件
        for i in range(len(coor_b)): # NO.2 - 对工况数循环

            # 坐标变换
            for h in xy_trans:

                new_nodes = np.zeros_like(ve_node)
                # 对每个节点应用变换
                for nn in range(len(ve_node)):
                    x, y = ve_node[nn]
                    new_nodes[nn] = h(x, y)

                # VEM json 文件
                json_name_vem = f'vem-no{j_count}-{jname}'  # 为文件命名
                json_data_vem = os.path.join(vepath, json_name_vem) # 绝对路径
                write_dict(json_data_vem, new_nodes, ve_elem, material, coor_b[i], disp_b[i], n_bc_vem[i])

                # FEM json 文件
                json_name_fem = f'fem-no{j_count}-{jname}'  # 为文件命名
                json_data_fem = os.path.join(fepath, json_name_fem) # 绝对路径
                write_dict(json_data_fem, fe_node, fe_elem, material, coor_b[i], disp_b[i], n_bc_fem[i])
                j_count += 1
    
    return j_count


def generate_quad_mesh(n):
    """
    生成二维结构化四边形网格
    二维梁
    
    参数:
    n: int - 每条边的离散单元数量
    
    返回:
    nodes: ndarray - 节点坐标数组，形状为((n+1)*(n+1), 2)
    elements: ndarray - 单元节点索引数组，形状为(n*n, 4), 索引从1开始
    """
    # 计算节点总数
    n_y = n
    n_x = 10*n    # 悬臂梁x方向与y方向10倍离散
    num_nodes = (n_y + 1) * (n_x + 1)
    
    # 生成节点坐标
    x = np.linspace(0, 10, n_x + 1)
    y = np.linspace(0, 1, n_y + 1)
    
    # 创建网格点
    X, Y = np.meshgrid(x, y)
    
    # 将网格点展平并组合成节点坐标数组
    nodes = np.vstack((X.flatten(), Y.flatten())).T
    
    # 生成单元节点索引
    elements = []
    for j in range(n_y):
        for i in range(n_x):
            # 计算当前单元的四个节点索引
            # 注意: 节点编号从1开始
            n1 = j * (n_x + 1) + i + 1
            n2 = j * (n_x + 1) + i + 1 + 1
            n3 = (j + 1) * (n_x + 1) + i + 1 + 1
            n4 = (j + 1) * (n_x + 1) + i + 1
            elements.append([n1, n2, n3, n4])
    
    # 转换为numpy数组
    elements = np.array(elements, dtype=np.int64)
    
    return nodes, elements


def generate_bound_cond(value_u):
    """
    创建本质边界条件
    """
    n_cc = 1  # 自定义工况数
    coor_b = np.zeros((n_cc, 4, 2))
    disp_b = np.zeros((n_cc, 4, 2))

    # s0
    # 悬臂梁 - 侧边固支
    # Essential boundary condition: 1-y, 0-n
    x_00, x_01, x_10, x_11 = 1, 1, 0, 0
    y_00, y_01, y_10, y_11 = 0, 0, 0, 0
    coor_b[0] = [[x_00, x_01], [x_10, x_11], [y_00, y_01], [y_10, y_11]]
    # Displacement on essential boundary condition
    ux_00, ux_01, ux_10, ux_11 = 0, 0, 0, 0
    uy_00, uy_01, uy_10, uy_11 = 0, 0, 0, 0
    disp_b[0] = [[ux_00, ux_01], [ux_10, ux_11], [uy_00, uy_01], [uy_10, uy_11]]

    return coor_b, disp_b


def generate_f_cond(node, f_x, f_y):
    """
    创建面力边界条件
    """
    # 找到y=1直线上的点（带容差）
    mask = np.abs(node[:, 1] - 1) < 1e-6
    indices = np.where(mask)[0]
    points = node[mask]
    
    # 按x坐标排序并保持索引对应
    sorted_idx = points[:, 0].argsort()
    sorted_indices = indices[sorted_idx]
    sorted_points = points[sorted_idx]
    
    # 创建字典：索引-坐标
    node_dict = dict(zip(sorted_indices, sorted_points))
    
    # 计算相邻节点距离（基于排序后的坐标）
    if len(sorted_points) > 1:
        distances = np.linalg.norm(sorted_points[1:] - sorted_points[:-1], axis=1)
    else:
        distances = np.array([])
        print('!!!WARNING: 为搜索到y==1边界上的点')
    
    e_bc = []
    for i in range(len(f_x)): # 对工况循环
        f_value_x, f_value_y = f_x[i], f_y[i]
        ebc_i = np.zeros((6, len(sorted_indices)-1))
        
        for j in range(len(sorted_indices)-1): # 对边界点
            p_id1, p_id2 = sorted_indices[j] + 1, sorted_indices[j+1] + 1
            p_dis = 0.5*distances[j]
            ebc_i[0:2, j] = [p_id1, p_id2]
            # ebc_i[2:, j] = [p_dis*f_value_x, p_dis*f_value_y, p_dis*f_value_x, p_dis*f_value_y]
            ebc_i[2:, j] = [f_value_x, f_value_y, f_value_x, f_value_y]
        
        e_bc.append(ebc_i)
    
    return e_bc
    


def convert_to_serializable(data):
    """将数据转换为 JSON 可序列化的格式"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data


def write_dict(fpath, node, elem, material, xy_b, u_b, n_bc):
    """
    创建fdata字典
    输入： node, elem, material, disp_bc
    输出： fdata
    """
    fdata = {"Title": fpath}
    # dof
    fdata["nsd"] = 2
    fdata["ndof"] = 2
    fdata["nnp"] = len(node)
    fdata["nel"] = len(elem)
    fdata["nen"] = 4
    ven = np.zeros(len(elem))
    for i in range(len(elem)):
        ven[i] = np.size(elem[i])
    fdata["ven"] = ven

    # Material
    # fdata["E"] = 10
    # fdata["nu"] = 0.25
    fdata["E"] = material[0]
    fdata["nu"] = material[1]

    # FEM guass point
    fdata["ngp"] = 1

    # VEM stab variable
    fdata["stab_var"] = 1.0

    flags, e_bc, nd = IdentiBound(node, xy_b, u_b)
    fdata["flags"] = flags
    fdata["e_bc"] = e_bc
    fdata["nd"] = nd

    fdata["nbe"] = n_bc.shape[1]
    fdata["n_bc"] = n_bc

    # Plane stress or strain
    fdata["plane_strain"] = 0

    # Post process
    fdata["plot_mesh"] = "yes"
    fdata["plot_nod"] = "yes"
    fdata["plot_disp"] = "yes"
    fdata["print_disp"] = "no"
    fdata["compute_stress"] = "no"
    fdata["plot_stress_xx"] = "yes"
    fdata["plot_mises"] = "yes"
    fdata["plot_tex"] = "no"
    fdata["fact"] = 1

    # mesh
    fdata["x"] = node[:, 0]
    fdata["y"] = node[:, 1]
    fdata["IEN"] = elem

    fdata["y_cut"] = [0.5]

    # print(fdata)

    # Write in json file
    # 转换所有 NumPy 数组为列表
    """
    fdata["ven"] = fdata["ven"].tolist()
    fdata["flags"] = fdata["flags"].tolist()
    fdata["e_bc"] = fdata["e_bc"].tolist()
    fdata["x"] = fdata["x"].tolist()
    fdata["y"] = fdata["y"].tolist()
    """
    fdata = convert_to_serializable(fdata)

    # 生成json文件
    with open(fpath, 'w') as f_obj:
        json.dump(fdata, f_obj, indent=4)