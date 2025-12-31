"""
read the '.mat' mesh file
write the '.json' model file for VE or FE Solvers
"""
import numpy as np
import json
from scipy.io import loadmat

def IdentiBound(node, xy_b, u_b):
    # 提取x和y坐标
    x = node[:, 0]
    y = node[:, 1]
    
    # 计算边界值
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    
    # 找出边界上的节点索引
    x_ind_0 = np.where((x >= x_min-1e-5) & (x <= x_min+1e-5))[0].tolist()
    x_ind_1 = np.where((x >= x_max-1e-5) & (x <= x_max+1e-5))[0].tolist()
    y_ind_0 = np.where((y >= y_min-1e-5) & (y <= y_min+1e-5))[0].tolist()
    y_ind_1 = np.where((y >= y_max-1e-5) & (y <= y_max+1e-5))[0].tolist()

    b_ind = [x_ind_0, x_ind_1, y_ind_0, y_ind_1]

    # 判断节点自由度对应的边界条件
    flags = np.zeros(2*len(node))
    e_bc = np.zeros(2*len(node))
    nd = 0
    for i in range(len(node)):
        # dof 0
        for j in range(4):
            if (xy_b[j][0] and i in b_ind[j]):
                flags[2*i] = 2
                e_bc[2*i] = u_b[j][0]
                nd += 1
                break

        # dof 1
        for j in range(4):
            if (xy_b[j][1] and i in b_ind[j]):
                flags[2*i+1] = 2
                e_bc[2*i+1] = u_b[j][1]
                nd += 1
                break
            
    return flags, e_bc, nd


def convert_to_2d_array(data):
    """
    将嵌套数组结构转换为二维列表
    """
    result = []
    for item in data:
        # 提取最内层的数组
        inner_array = item[0][0]
        # 转换为Python列表并添加到结果中
        result.append(inner_array.tolist())
    return result

# ==================================================
# ====================** main **====================
# ==================================================
# Input mesh
if __name__ == "__main__":
    
    mpath = r"D:/bo4_2025/MLcode-test/FEM-converge/mesh/Rec20m20.mat"
    mdata = loadmat(mpath)

    node = mdata['node']
    elem = mdata['elem']
    elem = convert_to_2d_array(elem)

    # Create json file & Define variables
    # path
    fpath = "D:/bo4_2025/MLcode-test/FEM-converge/mesh/Rec20m20.json"
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
    fdata["E"] = 10
    fdata["nu"] = 0.25

    # FEM guass point
    fdata["ngp"] = 1

    # VEM stab variable
    fdata["stab_var"] = 0.5

    # Boundary condition
    # Essential boundary condition: 1-y, 0-n
    x_00, x_01, x_10, x_11 = 0, 0, 0, 0
    y_00, y_01, y_10, y_11 = 1, 1, 1, 0
    xy_b = [[x_00, x_01], [x_10, x_11], [y_00, y_01], [y_10, y_11]]

    # Displacement on essential boundary condition
    ux_00, ux_01, ux_10, ux_11 = 0, 0, 0, 0
    uy_00, uy_01, uy_10, uy_11 = 0, 0, 0.2, 0
    u_b = [[ux_00, ux_01], [ux_10, ux_11], [uy_00, uy_01], [uy_10, uy_11]]

    flags, e_bc, nd = IdentiBound(node, xy_b, u_b)
    fdata["flags"] = flags
    fdata["e_bc"] = e_bc
    fdata["nd"] = nd

    fdata["nbe"] = 0

    # Plane stress or strain
    fdata["plane_strain"] = 0

    # Post process
    fdata["plot_mesh"] = "yes"
    fdata["plot_nod"] = "yes"
    fdata["plot_disp"] = "yes"
    fdata["print_disp"] = "yes"
    fdata["compute_stress"] = "yes"
    fdata["plot_stress_xx"] = "no"
    fdata["plot_mises"] = "no"
    fdata["plot_tex"] = "no"
    fdata["fact"] = 1

    # mesh
    fdata["x"] = node[:, 0]
    fdata["y"] = node[:, 1]
    fdata["IEN"] = elem

    print(fdata)

    # Write in json file
    # 转换所有 NumPy 数组为列表
    fdata["ven"] = fdata["ven"].tolist()
    fdata["flags"] = fdata["flags"].tolist()
    fdata["e_bc"] = fdata["e_bc"].tolist()
    fdata["x"] = fdata["x"].tolist()
    fdata["y"] = fdata["y"].tolist()

    with open(fpath, 'w') as f_obj:
        json.dump(fdata, f_obj, indent=4)



