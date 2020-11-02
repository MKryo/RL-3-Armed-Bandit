import random
import numpy as np
import matplotlib.pyplot as plt


#環境
p=[0.3,0.5,0.7] #グローバル変数

def draw(arm_index): #腕のインデックスを受け取って、その腕の確率に従って報酬を返す
  if random.uniform(0,1) > p[arm_index]:
    return 0
  else:
    return 1


#エージェント（本当はクラスにするべき）

epsilon = 0.3 #グローバル変数

#腕をε-greedy方策に従って腕を選択する関数
def select_arm():
  if random.uniform(0,1) > epsilon:
    #7割の確率でgreedy行動
    for i in range(3):
      if max(q) == q[i]:
        arm_index = i
    return arm_index
  else:
    #残り３割の確率で探索
    return random.randint(0,2) #randintは指定した引数の範囲からランダムに整数で返す関数
    
def update(arm_index, reward):#標本平均手法を使って推定価値qを更新
  arm_counts[arm_index]+=1
  q[arm_index] = q[arm_index] + (1/arm_counts[arm_index]*(reward - q[arm_index]))



#main
#二次元配列で初期化し、全部保存
regret = np.zeros((100, 1000))

for i in range(0,100):
  #エージェントのパラメータ初期化
  q = [0,0,0]#各腕の推定価値　この価値を更新していく
  arm_counts = [0,0,0]#各腕が何回選ばれたかのカウント
  step = np.zeros(1000)

  for t in range(0,1000):
    arm_index = select_arm()#エージェントによって腕を選択する
    reward = draw(arm_index)#選択された腕を環境に与え、その腕に関する報酬を返す
    update(arm_index, reward)#選択した腕と環境から返された報酬をエージェントに伝え、期待値の更新を行う
    step[t] += t

    #regretは累積（積み重なっていく）ので、一個前の報酬確率の差を加えていく
    regret[i][t] += p[2]-p[arm_index]
    if t != 0:
      regret[i][t] += regret[i][t-1]

print(regret.shape)

#二次元配列の列方向の平均をとる
regret = np.mean(regret, axis=0) #axis=0が列方向という意味の引数

print(regret.shape)
print(q)#各腕の最終的な推定価値
print(arm_counts)#各腕の最終的な選択回数



#結果の表示
plt.xlabel('step')
plt.ylabel('regret')
plt.plot(step, regret, label="ε-greedy ε=0.3")
plt.legend()
plt.show()
