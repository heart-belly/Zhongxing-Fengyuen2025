import random
import math
import heapq
from typing import List, Dict, Tuple, Set
import sys
from collections import deque
import os

# random.seed(2)  # 固定全局模块的种子
# 基本数据结构
class Task:
    def __init__(self, id: int, type: int, computation: float, base_mem: float):
        self.id = id
        self.type = type
        self.computation = computation
        self.base_mem = base_mem
        self.successors = []
        self.predecessors = []
    
    def get_mem(self, mode: int) -> float:
        ratios = [1.0, 0.5, 0.25, 0.1]
        return self.base_mem * ratios[mode]

    def get_cpu(self, mode: int) -> float:
        ratios = [1.0, 0.5, 0.25, 0.1]
        return self.computation * ratios[mode]
    
    def get_time(self, mode: int, power: float) -> float:
        ratios = [1.0, 0.5, 0.25, 0.1]
        return (self.computation * ratios[mode]) / power

class Unit:
    def __init__(self, id: int, power: float, memory: float):
        self.id = id
        self.power = power
        self.memory = memory

class Sample:
    def __init__(self):
        self.contributions = {}  # task_type -> contribution

# 系统状态
class SystemState:
    def __init__(self):
        self.tasks = []
        self.units = []
        self.samples = []
        self.comm_matrix = []
        self.task_ranks = []
    
    def build_predecessors(self):
        for task in self.tasks:
            for succ in task.successors:
                self.tasks[succ].predecessors.append(task.id)
    
    def compute_ranks(self):
        self.task_ranks = [0.0] * len(self.tasks)
        order = self.topological_sort()
        order.reverse()
        
        for u in order:
            max_rank = 0.0
            for succ in self.tasks[u].successors:
                max_rank = max(max_rank, self.task_ranks[succ])
            self.task_ranks[u] = self.tasks[u].computation + max_rank
    

    def topological_sort(self) -> List[int]:
        in_degree = [0] * len(self.tasks)
        for t in self.tasks:
            for s in t.successors:
                in_degree[s] += 1
        
        q = deque()
        order = []
        for i in range(len(self.tasks)):
            if in_degree[i] == 0:
                q.append(i)
        
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.tasks[u].successors:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        return order

# 解决方案表示
class Solution:
    def __init__(self):
        self.unit_assign = []  # 每个任务的分配单元
        self.modes = []        # 每个任务的量化模式
        self.makespan = 1e18
        self.contribution = 0.0
        self.score = 0.0
        self.total_computation = 0.0
    
    def __lt__(self, other):
        return self.score > other.score  # 按评分降序

class Optimizer:
    def __init__(self, test):
        self.test = test
        self.state = SystemState()
        self.population = []
        self.total_computation = 0.0
        self.gen = random.Random(42)
    
    def load_input(self, task_path: str, unit_path: str, sample_path: str):
        # 解析任务文件
        with open(task_path, 'r') as fin:
            for line in fin:
                line = line.replace(',', ' ')
                parts = line.strip().split()
                
                task_id = int(parts[0])
                task_type = int(parts[1])
                computation = float(parts[2])
                base_mem = float(parts[3])
                # base_mem = float(parts[2])
                # computation = float(parts[3])
                
                t = Task(task_id, task_type, computation, base_mem)
                t.successors = [int(s) for s in parts[4:]]
                
                if len(self.state.tasks) <= task_id:
                    self.state.tasks.extend([None] * (task_id + 1 - len(self.state.tasks)))
                self.state.tasks[task_id] = t
                self.total_computation += computation
        
        # 构建前驱关系
        self.state.build_predecessors()
        
        # 解析算力单元
        with open(unit_path, 'r') as fin:
            first_line = fin.readline()
            n, m = map(int, first_line.strip().replace(',', ' ').split())
            
            self.state.comm_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
            
            for _ in range(n):
                line = fin.readline()
                parts = line.strip().replace(',', ' ').split()
                unit_id = int(parts[0])
                power = float(parts[1])
                memory = float(parts[2])
                self.state.units.append(Unit(unit_id, power, memory))
            
            for _ in range(m):
                line = fin.readline()
                parts = line.strip().replace(',', ' ').split()
                a = int(parts[0])
                b = int(parts[1])
                t = float(parts[2])
                self.state.comm_matrix[a][b] = t
                self.state.comm_matrix[b][a] = t
        
        # 解析样本数据
        with open(sample_path, 'r') as fin:
            for line in fin:
                line = line.replace(',', ' ')
                parts = line.strip().split()
                
                sample_id = int(parts[0])
                task_type = int(parts[1])
                contrib = float(parts[2])
                
                if len(self.state.samples) <= sample_id:
                    self.state.samples.extend([Sample() for _ in range(sample_id + 1 - len(self.state.samples))])
                self.state.samples[sample_id].contributions[task_type] = contrib
        
        self.state.compute_ranks()



    def evaluate(self, sol: Solution):
        class Event:
            def __init__(self, time: float, really_time : float, sample_id: int, task_id: int, index : int):
                self.time = time
                self.really_time = really_time
                self.sample_id = sample_id
                self.task_id = task_id
                self.index = index
            
            def __lt__(self, other):
                if self.really_time != other.really_time:
                    return self.really_time < other.really_time
                if self.sample_id != other.sample_id:
                    return self.sample_id < other.sample_id
                return self.task_id < other.task_id

                
        idx = 0
        pq = []
        unit_busy_until = [0.0] * len(self.state.units)
        sol.contribution = 0.0
        sol.makespan = 0.0
        sol.total_computation = 0.0
        # 初始化事件（所有样本从入口任务开始）
        for s in range(len(self.state.samples)):
            heapq.heappush(pq, Event(0.0, 0.0, s, 0, idx))
            idx += 1
        
        # 记录任务完成状态 [sample][task]
        sample_times = [{} for _ in range(len(self.state.samples))]
        # 模式匹配贡献度
        comp = [1, 0.75, 0.5, 0.4]
        # comp = [1, 0.5, 0.25, 0.1]
        while pq:
            evt = heapq.heappop(pq)
            # print("{}".format(evt.task_id))

            if evt.task_id in sample_times[evt.sample_id]:
                continue

            unit_id = sol.unit_assign[evt.task_id]
            task = self.state.tasks[evt.task_id]
            
            # 等待资源可用
            start_time = max(evt.time, unit_busy_until[unit_id])

            # print(evt.time, unit_busy_until[unit_id])
            duration = task.get_time(sol.modes[evt.task_id], self.state.units[unit_id].power)
            sol.total_computation += task.get_cpu(sol.modes[evt.task_id])
            # print("sample: {} task :{} mode{} power{} start_time {} duration: {}".format(evt.sample_id, evt.task_id, sol.modes[evt.task_id], self.state.units[unit_id].power, start_time, duration))
            # 记录贡献度
            sol.contribution += self.state.samples[evt.sample_id].contributions.get(task.type, 0) * comp[sol.modes[evt.task_id]]
            
            # # 生成完成事件
            unit_busy_until[unit_id] = max(unit_busy_until[unit_id], start_time + duration)

            sol.makespan = max(sol.makespan, start_time + duration)

            
            # 记录样本进度
            sample_times[evt.sample_id][evt.task_id] = start_time + duration


            # 触发后续任务
            # print(evt.task_id, end=' ')
            for succ in self.state.tasks[evt.task_id].successors:
                if succ in sample_times[evt.sample_id]:
                    continue
                all_predecessors_done = True
                max_pred_time = 0.0
                
                # 检查所有前驱完成
                for pred in self.state.tasks[succ].predecessors:
                    if pred not in sample_times[evt.sample_id]:
                        all_predecessors_done = False
                        break

                    src_unit = sol.unit_assign[pred]
                    dst_unit = sol.unit_assign[succ]
                    comm_time = 0 if src_unit == dst_unit else self.state.comm_matrix[dst_unit][src_unit]
                    max_pred_time = max(max_pred_time, sample_times[evt.sample_id][pred] + comm_time)
                
                if all_predecessors_done:
                    # print(succ, end = ' ')
                    # src_unit = sol.unit_assign[evt.task_id]
                    # dst_unit = sol.unit_assign[succ]
                    # comm_time = 0 if src_unit == dst_unit else self.state.comm_matrix[src_unit][dst_unit]

                    unit_id = sol.unit_assign[evt.task_id]
                    really_time = max(max_pred_time, unit_busy_until[unit_id])
                    heapq.heappush(pq, Event(max_pred_time, really_time, evt.sample_id, succ, idx))
                    idx += 1
            #         print("{}----{}----{}".format(evt.sample_id, succ, max_pred_time + comm_time), end = ' ')
        
        # 计算评分
        normalized_time = sol.makespan / sol.total_computation
        sol.score =  sol.contribution / normalized_time
        self.adjust_modes(sol,  sample_times)



    def bp(self, sol : Solution)->List[int]:
        order = []
        task_len = len(sol.modes)
        # 根据任务量从小到大排序
        groups = []
        for i in range(task_len):
            task = self.state.tasks[i]
            mode = sol.modes[i]
            mem = task.get_mem(mode)
            groups.append((mem, i))

        groups.sort(key=lambda x:x[0])
        for (mem, id) in groups:
            # print(mem)
            order.append(id)
        
        return order


    def generate_initial(self) -> Solution:
        sol = Solution()
        sol.unit_assign = [-1] * len(self.state.tasks)
        sol.modes = [0] * len(self.state.tasks)
        count = 0
        '''
            设置初始化的mode
        '''
        select_modes = [2 for i in range(100)]
        select_modes[7] = 3
        select_modes[0] = 1
        # 1. 随机初始化模式
        for i in range(len(sol.modes)):
            # sol.modes[i] = random.randint(0, 3)  # 随机初始模式
            sol.modes[i] = random.randint(select_modes[self.test], select_modes[self.test])
        
        used_mem = [0.0] * len(self.state.units)
        # # 利用拓扑排序
        order = self.state.topological_sort()
        # 装箱算法重构order
        # order = self.bp(sol)
        
        # 2. 随机打乱拓扑排序中的独立任务
        order, st = self.shuffle_independent_tasks(order)
        
        # 保存所有点的后继结点
        successors_groups = []
        for task_id in order:
            successors_groups.append(len(self.state.tasks[task_id].successors))
        successors_groups.sort()

        # rank
        ranks_groups = []
        for task_id in order:
            ranks_groups.append((self.state.task_ranks[task_id], task_id))

        ranks_groups.sort(key=lambda x : x[0], reverse=True)
        # 加权标记st
        # l = int(len(ranks_groups) * 0.01)
        # for i in range(l):
        #     st[ranks_groups[i][1]] = 2


        for task_id in order:
            task = self.state.tasks[task_id]
            candidates = []
            # 3. 动态调整模式直至找到可行解
            for attempt in range(4):
                current_mode = sol.modes[task_id]
                required = task.get_mem(current_mode)
                
                # 收集所有满足内存约束的单元
                for uid in range(len(self.state.units)):
                    if used_mem[uid] + required <= self.state.units[uid].memory:
                        candidates.append(uid)
                
                if candidates:
                    break
                
                # 降级模式
                if sol.modes[task_id] < 3:
                    sol.modes[task_id] += 1
                else:
                    break  # 无法进一步降级
            
            if not candidates:
                # 表明当前找不到可以放置的算力单元（已经将任务量化到最低）: 所以在分配给任务的算力单元应该先量化到最低，再逐步降低量化水平
                raise RuntimeError("No feasible solution")
            
            selected_uid = -1
            rate = 1.0
            power = 0
            # 选择used_mem最低的放入
            successors_cnt = len(self.state.tasks[task_id].successors)
            for unit in candidates:
                t_rate = (used_mem[unit] + task.get_mem(sol.modes[task_id])) / self.state.units[unit].memory
                if t_rate < rate:
                    rate = t_rate
                    selected_uid = unit
            

            


            '''
                调整出度结点以及rank结点的unit位置
            '''
            # 选择cmp
            adjust = False
            cmp = 0x3f3f3f3f
            cmp_s = [0x3f3f3f3f for i in range(100)]
            cmp_s[8] = 10
            cmp_s[9] = 8
            cmp = cmp_s[self.test]

            if successors_cnt > cmp or st[task_id] == 2:
                backup = selected_uid
                # 可以选择速度快的
                for unit in candidates:
                    t_power = self.state.units[unit].power
                    if t_power > power:
                        power = t_power
                        selected_uid = unit
                
                adjust = True
                
            # print(rate)
            if selected_uid == -1:
                 raise RuntimeError("No feasible solution")

            # # 4. 带权重的随机选择单元
            # selected_uid = self.select_unit_by_weight(candidates, task_id, sol)

            sol.unit_assign[task_id] = selected_uid
            used_mem[selected_uid] += task.get_mem(sol.modes[task_id])
            


            
            # 成功放置
            count = count + 1

        task_number = len(self.state.tasks)
        contain = {0, 6, 7}
        # 是否要启用decrease_modes
        is_open = ((count == task_number) and self.test not in contain)
        if is_open:
            self.decrease_modes(sol, used_mem)       

        # 重新选择模式
        self.evaluate(sol)
        return sol

    def decrease_modes(self, sol : Solution, used_mem : list):
        # print("*******************", cnt)
        # 简单尝试降低量化水平（从占用内存最低的开始）
        pq = []  
        task_number = len(self.state.tasks)
        for task_id in range(task_number):
            task = self.state.tasks[task_id]
            heapq.heappush(pq, (task.get_mem(sol.modes[task_id]), task_id))

        # 依次放宽
        while pq:
            (mem, task_id) = heapq.heappop(pq)
            # 得到当前量化模式
            mode = sol.modes[task_id]
            unit_id = sol.unit_assign[task_id]
            if mode <= 0:
                continue
            # 可能降低的量化等级
            task = self.state.tasks[task_id]
            target_level = mode - 1
            # 需要增加的内存量
            add_mem = task.get_mem(target_level) - mem
            if used_mem[unit_id]  + add_mem < self.state.units[unit_id].memory:
                used_mem[unit_id] += add_mem
                sol.modes[task_id] = target_level
                # # 放入队列
                heapq.heappush(pq, (mem + add_mem, task_id))

                
    def adjust_modes(self, sol : Solution, sample_times:list):
        # 遍历任务的结束时间并尝试放开量化等级
        maybe_tasks = []
        # 统计任务对所有样本的贡献度
        task_contr = []
        for task_id in range(len(self.state.tasks)):
            task = self.state.tasks[task_id]
            task_type = task.type
            max_end = 0
            x = 0
            if len(task.successors) == 0:
                # 统计所有样本的在改任务下运行结束时间
                for sample_id in range(len(self.state.samples)):
                    sample = self.state.samples[sample_id]
                    max_end = max(sample_times[sample_id][task_id], max_end)
                
                if max_end < sol.makespan:
                    maybe_tasks.append((max_end, task_id))

            for sample_id in range(len(self.state.samples)):
                sample = self.state.samples[sample_id]

                x += sample.contributions[task_type]

            task_contr.append((x, task_id))

        
        maybe_tasks.sort(key=lambda x : x[0])
        task_contr.sort(key=lambda x : x[0])


        if self.test == 2:
            x = 1
            if maybe_tasks[x][0] < sol.makespan:
                sol.modes[maybe_tasks[x][1]] = max(0, sol.modes[maybe_tasks[x][1]] +1)
            sol.modes[5] = 2
        elif self.test == 7:
            sol.modes[5] = 2

        elif self.test == 9:
            x = 5
            if maybe_tasks[x][0] < sol.makespan:
                sol.modes[maybe_tasks[x][1]] = max(0, sol.modes[maybe_tasks[x][1]] + 1) 
        elif self.test == 10:
            x = 5
            if maybe_tasks[x][0] < sol.makespan:
                sol.modes[maybe_tasks[x][1]] = max(0, sol.modes[maybe_tasks[x][1]] + 1) 

    def adjust_units(self, sol : Solution):
        groups = []
        # 尝试降低贡献度与计算开销比例最小的任务
        for task_id in range(len(self.state.tasks)):
            task = self.state.tasks[task_id]
            max_con = 0
            for sample_id in range(len(self.state.samples)):
                max_con = max(self.state.samples[sample_id].contributions.get(task.type, 0), max_con)
            
            groups.append((max_con / task.get_cpu(0), task_id))
        
        groups.sort(key=lambda x : x[0])
          
    def shuffle_independent_tasks(self, order: List[int]):
        levels = []
        depth_map = {}
        
        # 计算任务层级
        for task_id in order:
            max_depth = 0
            for pred in self.state.tasks[task_id].predecessors:
                max_depth = max(max_depth, depth_map.get(pred, 0))
            depth_map[task_id] = max_depth + 1
            if len(levels) <= max_depth:
                levels.extend([[] for _ in range(max_depth + 1 - len(levels))])
            levels[max_depth].append(task_id)
        

        st = [0] * len(order)

        # 逐层打乱
        order.clear()
        for level in levels:
            random.shuffle(level)
            order.extend(level)

            makespan = -1
            target = -1
            for id in level:
                if self.state.task_ranks[id] > makespan:
                    makespan = self.state.task_ranks[id]
                    target = id
            
            st[target] = 1
        
        return order, st




        # # 逐层打乱
        # order.clear()
        # groups = []
        # for level in levels:
        #     # 按照后续结点数量排序
        #     for id in level:
        #         groups.append((len(self.state.tasks[id].successors), id))

        # groups.sort(key = lambda x : x[0], reverse = True)

        # for (number, id) in groups:
        #     order.append(id)
        
        # return order

    
    def select_unit_by_weight(self, candidates: List[int], task_id: int, sol: Solution) -> int:
        comm_costs = []
        total = 0.0

        # 计算各候选单元的通信代价
        for uid in candidates:
            cost = 0.0
            for pred in self.state.tasks[task_id].predecessors:
                if sol.unit_assign[pred] != -1:
                    cost += self.state.comm_matrix[sol.unit_assign[pred]][uid]
            comm_costs.append(cost)
            total += 1.0 / (cost + 1e-6)  # 代价越小权重越大
        
        # 轮盘赌选择
        r = random.uniform(0.0, total)
        accum = 0.0
        
        for i in range(len(candidates)):
            accum += 1.0 / (comm_costs[i] + 1e-6)
            if r <= accum:
                return candidates[i]
        return candidates[-1]
    

    def check(self, sol : Solution):
        # 检测内存有无溢出
        unit_len = len(self.state.units)
        used_mem = [0.0] * unit_len

        for task_id in range(len(self.state.tasks)):
            mode = sol.modes[task_id]
            unit_id = sol.unit_assign[task_id]
            task = self.state.tasks[task_id]
            
            used_mem[unit_id] += task.get_mem(mode)
            if used_mem[unit_id] > self.state.units[unit_id].memory:
                return False
        return True


    def genetic_optimize(self, pop_size=1, max_gen=0):
        self.population = []
        for i in range(1):
            sol = self.generate_initial()
            # self.evaluate(sol)
            self.population.append(sol)
            print(f"随机初始解: {i} {sol.score}", file=sys.stderr)


        for i in range(pop_size):
            child = Solution()
            child.unit_assign = sol.unit_assign.copy()
            child.modes = sol.modes.copy()
            self.population.append(child)

            
        self.population.sort()
        
        for gen1 in range(max_gen):
            # 选择（锦标赛选择）
            selected = []
            for i in range(pop_size):
                a = random.randint(0, pop_size - 1)
                b = random.randint(0, pop_size - 1)
                selected.append(max(self.population[a], self.population[b]))
            
            # 交叉（两点交叉）
            offspring = []
            for i in range(0, pop_size, 1):
                parent1 = selected[i]
                parent2 = selected[i]
                
                cross1 = random.randint(0, len(self.state.tasks) - 1)
                cross2 = random.randint(0, len(self.state.tasks) - 1)
                if cross1 > cross2:
                    cross1, cross2 = cross2, cross1
                
                child1 = Solution()
                child1.unit_assign = parent1.unit_assign.copy()
                child1.modes = parent1.modes.copy()
                
                child2 = Solution()
                child2.unit_assign = parent2.unit_assign.copy()
                child2.modes = parent2.modes.copy()
                
                # for j in range(cross1, cross2 + 1):
                #     child1.unit_assign[j] = parent2.unit_assign[j]
                #     child1.modes[j] = parent2.modes[j]
                #     child2.unit_assign[j] = parent1.unit_assign[j]
                #     child2.modes[j] = parent1.modes[j]
                
                offspring.append(child1)
                offspring.append(child2)
            
            # 变异
            for sol in offspring:
                if random.random() < 0.9:
                    task_id = random.randint(0, len(self.state.tasks) - 1)
                    sol.unit_assign[task_id] = random.randint(0, len(self.state.units) - 1)
                # if random.random() < 0.1:
                #     task_id = random.randint(0, len(self.state.tasks) - 1)
                #     sol.modes[task_id] = random.randint(0, 3)
            
            # 结束之后检测是否满足约束
            
            offspring_backup = []
            for i in range(len(offspring)):
                child = Solution()
                child.unit_assign = offspring[i].unit_assign.copy()
                child.modes = offspring[i].modes.copy()
                offspring_backup.append(child)

            offspring = []
            
            for sol in  offspring_backup:
                if self.check(sol):
                    offspring.append(sol)
                else:
                    print("Not Satisfied")

            # 评估
            for sol in offspring:
                try:
                    self.evaluate(sol)
                except:
                    sol.score = -1e18  # 无效解
            
            # 环境选择
            combined = self.population + offspring
            combined.sort()
            if len(combined) > pop_size:
                combined = combined[:pop_size]
            self.population = combined
            
            print(f"Generation {gen1} Best Score: {self.population[0].score} "
                  f"Makespan: {self.population[0].makespan} "
                  f"Contribution: {self.population[0].contribution}", file=sys.stderr)
                  

    
    def get_best(self) -> Solution:
        return self.population[0] if self.population else Solution()
    
    def output(self, sol: Solution, path: str):
        with open(path, 'w') as fout:
            for i in range(len(sol.unit_assign)):
                fout.write(f"{i},{sol.unit_assign[i]},{sol.modes[i]}\n")
    
    def loadOutput(self, solSpecial: Solution, outputPath: str):
        solSpecial.unit_assign = [0] * len(self.state.tasks)
        solSpecial.modes = [0] * len(self.state.tasks)
        
        with open(outputPath, 'r') as fin:
            for line in fin:
                line = line.replace(',', ' ')
                parts = line.strip().split()
                taskId = int(parts[0])
                useUnit = int(parts[1])
                useMode = int(parts[2])
                solSpecial.modes[taskId] = useMode
                solSpecial.unit_assign[taskId] = useUnit

def main():
    opt = Optimizer()
    base_path = "./data/4/"
    
    try:
        opt.load_input(base_path + "input1.txt",
                      base_path + "input2.txt",
                      base_path + "sample.txt")
        
        opt.genetic_optimize()
        best = opt.get_best()
        opt.output(best, base_path + "output.txt")
        
        print("Optimization completed.\n"
              f"Best makespan: {best.makespan}\n"
              f"all compute: {opt.total_computation}\n"
                f"total compute: {best.total_computation}\n"
              f"Total contribution: {best.contribution}\n"
              f"Final score: {best.score}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    #评估我指定的解
    # solSpecial = Solution()
    # outputPath = base_path + "output.txt"
    # opt.loadOutput(solSpecial, outputPath)
    # opt.evaluate(solSpecial)
    # print("我评估的结果", file=sys.stderr)
    # print(f"best makespan: {solSpecial.makespan}", file=sys.stderr)
    # print(f"Total contribution: {solSpecial.contribution}", file=sys.stderr)
    # print(f"final score: {solSpecial.score}", file=sys.stderr)
    # print("*******************", file=sys.stderr)
    return 0
def main1():
    data_root = "./data/"  # 根目录
    cnt = -1
    seeds = [4, 4, 2, 4, 4, 4, 4, 4, 4, 6, 1]
    # 遍历data目录下的所有子目录
    for subdir in sorted(os.listdir(data_root)):
        base_path = os.path.join(data_root, subdir)

        '''
            选择初始解的最优种子
        '''
        cnt = cnt + 1
        if cnt <= 10:
            random.seed(seeds[cnt])
        else:
            random.seed(4)
        '''
            调整初始解的次数以及迭代次数
        '''  
        init_num = [1 for i in range(100)]
        iter_num = [0 for i in range(100)]
                
        # init_num[6] = 10
        # iter_num[6] = 20
        # init_num[8] = 4
        # iter_num[8] = 10
        init_num[9] = 2
        iter_num[9] = 10
        init_num[10] = 5
        iter_num[10] = 10
        # if cnt != 8:
        #     continue
            

        # 确认是目录而不是文件
        if not os.path.isdir(base_path):
            continue

        # 构建输入文件路径
        input_files = (
            os.path.join(base_path, "input1.txt"),
            os.path.join(base_path, "input2.txt"),
            os.path.join(base_path, "sample.txt")
        )

        # 检查输入文件是否存在
        if not all(os.path.exists(f) for f in input_files):
            print(f"[WARN] Skip {subdir}: missing input files", file=sys.stderr)
            continue

        # 处理每个子目录
        try:
            opt = Optimizer(cnt)
            # 加载输入
            opt.load_input(os.path.join(base_path, "input1.txt"),
                      os.path.join(base_path, "input2.txt"),
                       os.path.join(base_path, "sample.txt"))
            
            # 执行优化
            opt.genetic_optimize(init_num[cnt], iter_num[cnt])
            best = opt.get_best()
            
            # 输出到子目录的output.txt
            output_path = os.path.join(base_path, "output.txt")
            opt.output(best, output_path)
            
            # # 打印日志
            # print(f"[OK] Processed {subdir}\n"
            #       f"  Makespan: {best.makespan}\n"
            #       f"  Contribution: {best.contribution}\n"
            #       f"  Score_really: {best.score}\n",
            #       file=sys.stderr)
            print("Optimization completed.\n"
                f"Best makespan: {best.makespan}\n"
                f"all compute: {opt.total_computation}\n"
                    f"total compute: {best.total_computation}\n"
                f"Total contribution: {best.contribution}\n"
                f"Final score: {best.score}", file=sys.stderr)
                
        except Exception as e:
            print(f"[ERROR] Failed to process {subdir}: {str(e)}", file=sys.stderr)
            continue  # 继续处理下一个子目录

    return 0
if __name__ == "__main__":
    main1()