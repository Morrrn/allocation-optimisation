import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math
import os
import csv

# -----------------------------导入数据-----------------------------

df_merge_target = pd.read_csv('./data/merge_target.csv')
df_skc_merge_strategy = pd.read_csv('./data/skc_merge_strategy.csv')
df_store_product = pd.read_csv('./data/store_product.csv')

# 定义集合及字典，进行数据处理
# I表示商店的集合，J表示SKC的集合，K表示各SKC的尺码的集合
I = list(df_merge_target['warehouse_code'].unique())
J = list(df_merge_target['product_code'].unique())
K = list(df_merge_target['size_code'].unique())
# y: i店j号SKC的k尺码的期望数量
y = dict(zip(zip(df_merge_target['warehouse_code'], df_merge_target['product_code'],
                 df_merge_target['size_code']), df_merge_target['constrained_target_inv_qty']))
# a: i店j号SKC的k尺码的可用数量
a = dict(zip(zip(df_merge_target['warehouse_code'], df_merge_target['product_code'],
                 df_merge_target['size_code']), df_merge_target['available_inv_qty']))
# l: i店j号SKC的铺货标准下限
l = dict(zip(zip(df_merge_target['warehouse_code'], df_merge_target['product_code'], df_merge_target['size_code']),
             df_merge_target['inventory_lower_limit']))
# 若为空，则不考虑下限，将设为0
for key in l.keys():
    if math.isnan(l[key]):
        l[key] = 0
# u: i店j号SKC的铺货标准上限
u = dict(zip(zip(df_merge_target['warehouse_code'], df_merge_target['product_code'], df_merge_target['size_code']),
             df_merge_target['inventory_upper_limit']))
# 若为空，则不考虑上限，将设为无限大
for key in u.keys():
    if math.isnan(u[key]):
        u[key] = 100
# p: i店j号SKC的优先级
p = dict(zip(zip(df_store_product['warehouse_code'], df_store_product['product_code']),
             df_store_product['in_store_priority']))
# s: j号SKC归入店铺的上限数量
s = dict(zip(df_skc_merge_strategy['product_code'], df_skc_merge_strategy['store_upper_limit']))
# c: j号SKC在归出店里是否要调空
c = dict(zip(df_skc_merge_strategy['product_code'], df_skc_merge_strategy['out_store_clear_out']))
# w: 4个目标的权重
w = [0.25, 0.25, 0.25, 0.25]
# w = [0.3, 0.3, 0.3, 0.1]
# M是一个极大的数，用于辅助建立约束
M = GRB.INFINITY
# index用于构建变量b_2的索引，形如(i,j,k,i)
index = []
for k in a.keys():
    elem = k[0]
    item = k + (elem,)
    index.append(item)

# -----------------------------建立模型-----------------------------

model = gp.Model('Allocation')

# 建立变量
# x: 表示i店j号SKC的k尺码分配后的数量
x = model.addVars(a.keys(), vtype=GRB.INTEGER)
# 辅助约束2
b = model.addVars(a.keys(), vtype=GRB.BINARY)
z = model.addVars(a.keys(), lb=-100, ub=100, vtype=GRB.INTEGER)
# 辅助约束4
b_2 = model.addVars(index, vtype=GRB.BINARY)
z_2 = model.addVars(a.keys(), vtype=GRB.INTEGER)
# 辅助目标2
# z_3: 接收最大约束后的判断结果
z_3 = model.addVars(l.keys(), vtype=GRB.INTEGER)
# lx, xu分别接收l-x和x-u
lx = model.addVars(l.keys(), lb=-1000, vtype=GRB.INTEGER)
xu = model.addVars(l.keys(), lb=-1000, vtype=GRB.INTEGER)
# 辅助目标3
# tmp_abs: 用来接收目标3中的绝对值部分
t = model.addVars(a.keys(), lb=0, vtype=GRB.INTEGER)

# -------------添加约束-------------

# 约束1：数量平衡
model.addConstrs((gp.quicksum(x[i, j, k] for i in I if (i, j, k) in x) ==
                  gp.quicksum(a[i, j, k] for i in I if (i, j, k) in a)
                  for j in J for k in K), name='constrain_1')

# 约束2：调出店需要调空
model.addConstrs((x[i, j, k] <= a[i, j, k] for i in I for j in J for k in K
                  if (i, j, k) in x), name='constrain_2_1')
model.addConstrs((b[i, j, k] <= x[i, j, k] for i in I for j in J for k in K
                  if (i, j, k) in x), name='constrain_2_2')
model.addConstrs((z[i, j, k] >= x[i, j, k] - M * (1 - b[i, j, k])
                  for i in I for j in J for k in K if (i, j, k) in x), name='constrain_2_3')
model.addConstrs((z[i, j, k] <= M * b[i, j, k] for i in I for j in J for k in K
                  if (i, j, k) in x), name='constrain_2_4')
model.addConstrs((z[i, j, k] <= a[i, j, k] for i in I for j in J for k in K
                  if (i, j, k) in x and c[j] == 1), name='constrain_2_5')

# 约束3：对于调入店数目上限有要求
model.addConstrs((gp.quicksum(gp.min_(1, x[i, j, k]) for i in I if (i, j, k) in x) <= s[j]
                  for j in J for k in K if (j, k) in s), name='constrain_3')

# 约束4：在单尺码上，低优先级上的门店都归完了再归出高优先级的门店
# model.addConstrs((x[i, j, k] <= a[i, j, k] for i in I for j in J for k in K
#                   if (i, j, k) in x), name='constrain_4_1')
model.addConstrs((p[i_, j] - p[i, j] - M * (1 - b_2[i, j, k, i_]) <= 0
                  for i in I for j in J for k in K for i_ in I if (i, j, k, i_) in b_2), name='constrain_4_2')
model.addConstrs((x[i_, j, k] - a[i_, j, k] + M * (1 - b_2[i, j, k, i_]) >= 0
                  for i in I for j in J for k in K for i_ in I if (i, j, k, i_) in b_2), name='constrain_4_3')
model.addConstrs((z_2[i, j, k] == gp.quicksum(b_2[i, j, k, i_] for i_ in I if (i, j, k, i_) in b_2)
                  for i in I for j in J for k in K if (i, j, k) in x), name='constrain_4_4')
model.addConstrs((z_2[i, j, k] >= 1 for i in I for j in J for k in K
                  if (i, j, k) in x), name='constrain_4_5')

# 辅助目标2
model.addConstrs(lx[i, j, k] == l[i, j, k] - x[i, j, k]
                 for i in I for j in J for k in K if (i, j, k) in x)
model.addConstrs(xu[i, j, k] == x[i, j, k] - u[i, j, k]
                 for i in I for j in J for k in K if (i, j, k) in x)
for (i, j, k) in lx:
    model.addGenConstrMax(z_3[i, j, k], [0, lx[i, j, k], xu[i, j, k]])

# 辅助目标3
for (i, j, k) in x:
    model.addConstr(t[i, j, k] >= x[i, j, k] - y[i, j, k], name='constrain_abs_1')
    model.addConstr(t[i, j, k] >= y[i, j, k] - x[i, j, k], name='constrain_abs_2')

# -------------添加目标函数-------------

model.setObjective(w[0] * gp.quicksum(p[i, j] * x[i, j, k]
                                      for i in I for j in J for k in K if (i, j, k) in x) +
                   w[1] * gp.quicksum(z_3[i, j, k] for i in I for j in J if (i, j, k) in z_3) +
                   w[2] * gp.quicksum(t[i, j, k]
                                      for i in I for j in J for k in K if (i, j, k) in x) +
                   w[3] * gp.quicksum(p[i, j] * (x[i, j, k] - a[i, j, k])
                                      for i in I for j in J for k in K if (i, j, k) in x),
                   sense=GRB.MINIMIZE)

# -------------求解模型-------------

model.optimize()

# -----------------------------输出结果-----------------------------

output = [['warehouse_code', 'product_code', 'size_code', 'allocation_inv_qty']]
if model.status == gp.GRB.OPTIMAL:
    print("最优目标值为：", model.objVal)
    print("最优分配方案为：")
    for (i, j, k) in x:
        if x[i, j, k].x > 0:
            print(f"店铺 {i} 在 SKC {j} 下的尺码 {k} 的分配数量为 {x[i, j, k].x}")
            output.append([i, j, k, x[i, j, k].x])
else:
    print("模型无可行解或无最优解")

# -----------------------------保存结果-----------------------------

output_file_loc = 'result'
output_file_name = 'result.csv'
output_file_path = os.path.join(output_file_loc, output_file_name)

with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    for row in output:
        writer.writerow(row)

pass
