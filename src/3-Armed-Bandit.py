import random
import numpy as np
import matplotlib.pyplot as plt


class Environment:
  def __init__(self, prob): #コンストラクタ：インスタンス生成時に自動的に実行される関数
    self.p=prob

  def draw(self,arm_index): #腕のインデックスを受け取って、その腕の確率に従って報酬を返す
    if random.uniform(0,1) > self.p[arm_index]:
      return 0
    else:
      return 1

class Agent:
  def __init__(self, epsilon):#インスタンス生成時に自動的に実行される関数
    self.epsilon = epsilon

  #腕をε-greedy方策に従って腕を選択する関数
  def select_arm(self):
    if random.uniform(0,1) > self.epsilon:
      #1-εの確率でgreedy行動
      for i in range(3):
        if max(q) == q[i]:
          arm_index = i
      return arm_index
    else:
      #εの確率で探索
      return random.randint(0,2) #randintは指定した引数の範囲からランダムに整数で返す関数
      
  def update(self, arm_index, reward):#標本平均手法を使って推定価値qを更新
    arm_counts[arm_index]+=1
    q[arm_index] = q[arm_index] + (1/arm_counts[arm_index]*(reward - q[arm_index]))

#インスタンスの生成
#環境生成　　パラメータ：各腕の報酬確率
Env = Environment([0.3, 0.5, 0.7])
#エージェント生成　　パラメータ：ε
Agent = Agent(0.2)

#学習回数の指定
simulation_size = 100
step_size = 1000

#二次元配列で初期化し、全部保存
regret = np.zeros((simulation_size, step_size))


#main
for i in range(simulation_size):
  #エージェントのパラメータ初期化
  q = [0,0,0]#各腕の推定価値　この価値を更新していく
  arm_counts = [0,0,0]#各腕が何回選ばれたかのカウント

  for t in range(step_size):
    arm_index = Agent.select_arm()#エージェントによって腕を選択する
    reward = Env.draw(arm_index)#選択された腕を環境に与え、その腕に関する報酬を返す
    Agent.update(arm_index, reward)#選択した腕と環境から返された報酬をエージェントに伝え、期待値の更新を行う

    #regretは累積（積み重なっていくので）、一個前の報酬確率の差を加えていく
    regret[i][t] += Env.p[2]-Env.p[arm_index]
    if t != 0:
      regret[i][t] += regret[i][t-1]

print(regret.shape)
#二次元配列の列方向の平均をとる
regret = np.mean(regret, axis=0) #axis=0:列方向への操作  axis=1:行方向への操作
print(regret.shape)


#結果の表示
plt.xlabel('step')
plt.ylabel('regret')
plt.plot(regret, label="ε-greedy ε=0.3")
plt.legend()
plt.show()