from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.messages import AIMessage, ToolMessage
import json
from datetime import datetime
import sqlite3
from langchain.tools import BaseTool
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) 

llm = ChatOpenAI(
    temperature=0,
    model="THUDM/glm-4-9b-chat",  
    openai_api_key="sk-",
    openai_api_base="https://api.siliconflow.cn/v1",
    max_retries=0,
)


class SQLiteQueryTool(BaseTool):
    """SQLite查询自定义工具"""
    name: str = "sqlite_query"
    description: str = "执行SQLite数据库查询，支持SELECT语句查询数据。输入格式：'SQL查询语句|数据库路径'，例如：'SELECT * FROM emp LIMIT 3|10_nl2sql/mynl2sql/sample.db'"
    
    def _run(self, query_input: str) -> str:
        """执行SQLite查询"""
        try:
            # 解析输入：格式为 "SQL查询|数据库路径"
            if '|' not in query_input:
                return "输入格式错误，请使用格式：'SQL查询语句|数据库路径'"
            
            query, database_path = query_input.split('|', 1)
            query = query.strip()
            database_path = database_path.strip()
            
            # 检查数据库文件是否存在
            if not os.path.exists(database_path):
                return f"数据库文件不存在: {database_path}"
            
            # 安全检查：只允许SELECT语句
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return "出于安全考虑，只允许执行SELECT查询语句"
            
            # 连接数据库并执行查询
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            
            query = fix_sql_names(query)
            cursor.execute(query)
            results = cursor.fetchall()
            
            # 获取列名
            column_names = [description[0] for description in cursor.description]
            
            # 格式化结果
            if not results:
                result_str = "查询结果为空"
            else:
                # 构建表格式的结果
                result_str = "查询结果:\n"
                result_str += " | ".join(column_names) + "\n"
                result_str += "-" * (len(" | ".join(column_names))) + "\n"
                
                for row in results:
                    result_str += " | ".join(str(cell) for cell in row) + "\n"
                
                result_str += f"\n共返回 {len(results)} 条记录"
            
            cursor.close()
            conn.close()
            
            return result_str
            
        except sqlite3.Error as e:
            if "no such table: employees" in str(e):
                # 自动修正表名并重试
                query_fixed = fix_sql_names(query)
                try:
                    conn = sqlite3.connect(database_path)
                    cursor = conn.cursor()
                    cursor.execute(query_fixed)
                    results = cursor.fetchall()
                    # 获取列名
                    column_names = [description[0] for description in cursor.description]
                    # 格式化结果
                    if not results:
                        result_str = "查询结果为空"
                    else:
                        # 构建表格式的结果
                        result_str = "查询结果:\n"
                        result_str += " | ".join(column_names) + "\n"
                        result_str += "-" * (len(" | ".join(column_names))) + "\n"
                        
                        for row in results:
                            result_str += " | ".join(str(cell) for cell in row) + "\n"
                        
                        result_str += f"\n共返回 {len(results)} 条记录"
                    cursor.close()
                    conn.close()
                    return "（自动修正表名后重试成功）\n" + result_str
                except Exception as e2:
                    return f"自动修正表名后仍失败: {str(e2)}"
            return f"SQLite查询错误: {str(e)}"
        except Exception as e:
            return f"查询执行失败: {str(e)}"
    
    async def _arun(self, query_input: str) -> str:
        """异步执行查询"""
        return self._run(query_input)

tool = SQLiteQueryTool()

# 添加轨迹跟踪器
class AgentTracker:
    def __init__(self):
        self.trajectory = []
        self.current_step = 0
    
    def add_step(self, agent_name, action, content=None, timestamp=None):
        """添加轨迹步骤"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        step = {
            "step": self.current_step + 1,
            "agent": agent_name,
            "action": action,
            "content": content,
            "timestamp": timestamp
        }
        self.trajectory.append(step)
        self.current_step += 1
    
    def print_trajectory(self):
        """打印完整轨迹"""
        print("\n" + "="*80)
        print("🤖 AGENT 执行轨迹")
        print("="*80)
        for step in self.trajectory:
            print(f"步骤 {step['step']:2d} | {step['timestamp']} | {step['agent']:15s} | {step['action']}")
            if step['content']:
                print(f"       内容: {step['content'][:100]}{'...' if len(step['content']) > 100 else ''}")
        print("="*80)
    
    def get_trajectory_summary(self):
        """获取轨迹摘要"""
        agent_counts = {}
        for step in self.trajectory:
            agent = step['agent']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        summary = f"总步骤数: {len(self.trajectory)}, Agent调用次数: {agent_counts}"
        return summary

# 创建全局轨迹跟踪器
tracker = AgentTracker()

# 在AgentTracker类中增加任务进度跟踪
class TaskProgressTracker:
    def __init__(self):
        self.completed_tasks = set()
        self.total_tasks = 0
        self.task_list = []
    
    def add_task(self, task_description):
        """添加任务到列表"""
        self.task_list.append(task_description)
        self.total_tasks = len(self.task_list)
    
    def mark_completed(self, task_description):
        """标记任务为已完成"""
        self.completed_tasks.add(task_description)
    
    def get_progress(self):
        """获取任务进度"""
        return {
            "completed": len(self.completed_tasks),
            "total": self.total_tasks,
            "remaining": self.total_tasks - len(self.completed_tasks),
            "completed_tasks": list(self.completed_tasks),
            "remaining_tasks": [task for task in self.task_list if task not in self.completed_tasks]
        }

# 在主函数中初始化任务跟踪器
task_tracker = TaskProgressTracker()

def fix_sql_names(sql: str) -> str:
    """
    自动将常见英文表名/字段名替换为实际数据库中的表名/字段名。
    你可以根据实际表结构继续补充替换规则。
    """
    replacements = {
        # 表名替换
        "employees": "emp",
        "Employees": "emp",
        "EMPLOYEES": "emp",
        # 字段名替换
        "salary": "sal",
        "Salary": "sal",
        "SALARY": "sal",
        "department": "deptno",
        "Department": "deptno",
        "DEPARTMENT": "deptno",
        "position": "job",
        "Position": "job",
        "POSITION": "job",
    }
    for wrong, right in replacements.items():
        sql = sql.replace(wrong, right)
    return sql

def main():
    # 修改日志文件名，使用固定名称而不是时间戳
    log_file = f"{current_dir}/agent_log.log"
    report_file = f"{current_dir}/analysis_report.md"
    
    # 在开始新的运行前，清空日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"🚀 新的运行开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    # 清空分析报告文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 人力资源分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
    
    mainAgent = create_react_agent(
        llm,
        [
            create_handoff_tool(agent_name="intentAgent", description="转给【intentAgent】，它负责意图解析和任务拆解。"),
            create_handoff_tool(agent_name="sqlGenerateAgent", description="转给【sqlGenerateAgent】，它负责生成和处理SQL语句。")
        ],
        prompt="""你是 mainAgent，总工，负责与用户直接交互，并根据用户需求进行任务调度和分发。你的工作流程如下：
                1. 收到用户请求后，优先尝试直接生成一条可执行的 SQL 查询语句。
                    你是专业的 SQL 查询助手，需要根据用户问题生成对应的 SQL 查询语句。请严格遵循以下要求：
                    - 仅返回 SQL 查询语句，不要输出任何额外解释或说明。
                    - 充分参考以下数据库表结构信息，确保 SQL 语句的正确性：

                    数据库表结构信息：
                    - emp表：员工信息表，字段包括：empno(员工编号), ename(员工姓名), job(职位), mgr(上级编号), hiredate(入职日期), sal(工资), comm(奖金), deptno(部门编号)
                    - dept表：部门信息表，字段包括：deptno(部门编号), dname(部门名称), loc(部门位置)
                    - salgrade表：工资等级表，字段包括：grade(等级), losal(最低工资), hisal(最高工资)

                2. 如果能够直接生成 SQL 语句，则将任务转交给【sqlGenerateAgent】处理。
                3. 如果无法直接生成 SQL 语句，则将任务转交给【intentAgent】，由其进行意图解析和任务拆解。
                4. 你可以与【intentAgent】和【sqlGenerateAgent】多轮协作，确保任务顺利完成。
                5. 始终以高效、准确为目标，合理分配任务，提升整体协作效率。
                6. 只有在所有子任务都完成后，才回复"COMPLETE"，否则继续处理。

                请根据上述流程，智能判断并分发每一个用户请求，确保所有子任务都完成。""",
        name="mainAgent",
    )

    intentAgent = create_react_agent(
        llm,
        [
            create_handoff_tool(agent_name="sqlGenerateAgent", description="转给【sqlGenerateAgent】，它可以生成SQL语句。"),
            create_handoff_tool(agent_name="mainAgent", description="转给【mainAgent】，它可以转人工处理。")
        ],
        prompt="""你是 intentAgent，负责对用户复杂意图进行专业解析和任务拆解。你的工作流程如下：
                1. 在接收到任务后，首先分析用户的真实意图，并将复杂意图拆解为可执行的子任务。
                2. 对于每一个子任务，最多尝试3次完成。如果子任务可以直接生成 SQL 语句，则将任务转交给【sqlGenerateAgent】处理。
                3. 如果连续3次尝试都未能完成，或遇到无法解析、需要进一步澄清或需要主调度/人工介入的情况，则将任务转交给【mainAgent】。
                4. 你可以与【sqlGenerateAgent】和【mainAgent】进行多轮交互，确保任务顺利完成。
                5. 始终以高效、准确为目标，合理分配任务，提升整体协作效率。
                6. 只有在所有子任务都完成后，才回复"COMPLETE"，否则继续处理。

                请严格按照上述流程，智能判断并分发每一个任务，确保所有子任务都完成。""",
        name="intentAgent",
    )

    sqlGenerateAgent = create_react_agent(
        llm,
        [
            create_handoff_tool(agent_name="reportAgent", description="转给【reportAgent】，它可以执行SQL语句，并生成报表。"),
            create_handoff_tool(agent_name="mainAgent", description="转给【mainAgent】，它可以转人工处理。")
        ],
        prompt="""你是 sqlGenerateAgent，负责根据任务生成可执行的 SQL 语句。你的工作流程如下：
                1. 在接收到任务后，分析需求并尝试生成 SQL 语句，最多尝试3次。
                    你是专业的 SQL 查询助手，需要根据用户问题生成对应的 SQL 查询语句。请严格遵循以下要求：
                    - 仅返回 SQL 查询语句，不要输出任何额外解释或说明。
                    - 充分参考以下数据库表结构信息，确保 SQL 语句的正确性：

                    数据库表结构信息：
                    - emp表：员工信息表，字段包括：empno(员工编号), ename(员工姓名), job(职位), mgr(上级编号), hiredate(入职日期), sal(工资), comm(奖金), deptno(部门编号)
                    - dept表：部门信息表，字段包括：deptno(部门编号), dname(部门名称), loc(部门位置)
                    - salgrade表：工资等级表，字段包括：grade(等级), losal(最低工资), hisal(最高工资)

                2. 如果在3次尝试内成功生成 SQL 语句，则将其转交给【reportAgent】执行。
                3. 如果连续3次尝试都未能成功生成 SQL，或遇到无法处理的异常，请将任务转交给【mainAgent】进行人工处理。
                4. 只有在所有SQL语句都生成并执行完成后，才回复"COMPLETE"，否则继续处理。

                请严格按照上述流程完成每一个任务，确保所有SQL语句都生成并执行完成。""",
        name="sqlGenerateAgent",
    )

    reportAgent = create_react_agent(
        llm,
        [
            tool,
            create_handoff_tool(
                agent_name="mainAgent",
                description="转给【mainAgent】，用于汇报处理结果或遇到异常时回流主调度。"
            )
        ],
        prompt="""你是 reportAgent，专门负责执行SQL查询并生成分析报告。

## 工作流程

### 1. SQL执行阶段
- 收到SQL语句后，立即使用 sqlite_query 工具执行
- 输入格式：'SQL查询语句|10_nl2sql/mynl2sql/sample.db'
- 示例：'SELECT * FROM emp LIMIT 5|10_nl2sql/mynl2sql/sample.db'

### 2. 多SQL处理
- 如果收到多个SQL语句，按顺序逐个执行
- 每个SQL执行完成后记录结果
- 最后将所有结果整合输出

### 3. 任务连续性检查
- 执行完当前SQL后，检查是否还有未完成的分析任务
- 如果发现任务不完整，主动请求下一个SQL查询
- 确保所有必要的分析都完成后再生成最终报告
- **当所有SQL子任务都完成后，必须handoff给 mainAgent，请其生成综合性分析总结！**

### 4. 结果输出格式
执行成功后，按以下格式输出：
```
📊 SQL查询结果报告

🔍 执行的SQL语句：
[SQL语句]

📈 查询结果：
[数据表格形式展示]

📊 统计信息：
- 记录总数：[数量]
- 字段数量：[数量]
- 其他相关统计：[如有]

🔄 任务状态：
- 已完成：[已完成的任务]
- 待完成：[待完成的任务]
```

### 5. 错误处理
- 执行失败时自动重试，最多3次
- 3次重试后仍失败，使用 handoff 工具转给【mainAgent】
- 完全无法处理时，仅回复"END"

## 重要原则
- 必须立即执行收到的SQL语句，不得跳过
- 严格按照输出格式展示结果
- 遇到异常及时转交，不要自行处理复杂问题
- 保持专业、清晰的报告风格
- **确保任务连续性，不遗漏任何分析步骤**

## 工具使用说明
当需要执行SQL查询时，请使用以下格式调用 sqlite_query 工具：
- 参数名：query_input
- 参数值：'SQL语句|数据库路径'
- 例如：query_input = 'SELECT * FROM emp LIMIT 5|10_nl2sql/mynl2sql/sample.db'""",
        name="reportAgent",
    )

    checkpointer = InMemorySaver()
    workflow = create_swarm(
        [mainAgent, intentAgent, sqlGenerateAgent, reportAgent],
        default_active_agent="mainAgent"
    )
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "1","recursion_limit": 50}}
    max_turns = 15
    
    # 记录用户输入
    user_query = "我需要分析公司的人力资源情况，包括各部门的员工分布、工资水平统计，以及找出哪些部门的平均工资超过了公司整体平均水平，同时还要考虑员工的职位分布情况，最后生成一份综合性的分析报告"
    
    tracker.add_step("用户", "输入查询", user_query)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"🔍 用户查询: {user_query}\n")
    
    # 初始化消息历史
    messages = [{"role": "user", "content": user_query}]
    
    # 修改终止条件，增加更严格的完成判断
    def is_task_complete(response_content, original_query):
        """更严格的任务完成判断"""
        # 检查是否包含所有必要的分析内容
        required_keywords = [
            "员工分布", "工资水平", "平均工资", "职位分布", "分析报告", "综合性分析总结", "结论"
        ]
        
        # 检查是否执行了足够的SQL查询
        sql_count = response_content.count("SELECT")
        
        # 检查是否生成了完整的报告格式
        has_report_format = "📊 SQL查询结果报告" in response_content
        
        # 只有当包含大部分关键词且执行了多个SQL查询时才认为完成
        keyword_match = sum(1 for kw in required_keywords if kw in response_content)
        
        return (keyword_match >= 4 and sql_count >= 3 and has_report_format)

    # 在主循环中增加任务进度检查
    for i in range(max_turns):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n🔄 第 {i+1} 轮执行...\n")
        
        try:
            response = app.invoke(
                {"messages": messages},
                config,
            )
            
            # 获取新的消息
            new_messages = response.get('messages', [])
            
            # 检查是否有新的消息
            if len(new_messages) > len(messages):
                # 获取最新的消息
                latest_message = new_messages[-1]
                
                # 检查是否是AIMessage且有内容
                if hasattr(latest_message, 'content') and latest_message.content:
                    final_response = latest_message.content.strip()
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"🤖 Agent响应: {final_response}\n")
                    
                    # 记录Agent轨迹
                    agent_name = getattr(latest_message, 'name', 'unknown')
                    tracker.add_step(agent_name, "响应", final_response[:100])
                    
                    # 检查是否包含分析报告并保存到markdown文件
                    if any(keyword in final_response for keyword in ["📊", "分析报告", "统计信息", "查询结果", "人力资源", "员工分布", "工资水平", "综合性分析总结", "结论"]):
                        # 追加到固定的分析报告文件
                        with open(report_file, 'a', encoding='utf-8') as f:
                            f.write(f"## 用户查询\n{user_query}\n\n")
                            f.write(f"## 分析结果\n{final_response}\n\n")
                            f.write("---\n\n")
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"📄 分析报告已追加到: {report_file}\n")
                    
                    # 更严格的终止条件
                    if final_response == "END":
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write("❌ 流程已终止\n")
                        break
                    
                    # 检查是否完成任务（包含完整报告）
                    if is_task_complete(final_response, user_query):
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write("✅ 任务完成，已生成完整报告\n")
                        break
                    
                    # 更新消息历史
                    messages = new_messages
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"📝 消息历史已更新，当前消息数量: {len(messages)}\n")
                else:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write("⚠️ 未收到有效响应内容\n")
                    
                    # 如果消息内容为空，尝试继续下一轮而不是直接退出
                    if len(new_messages) > len(messages):
                        messages = new_messages
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write("🔄 消息内容为空，但继续下一轮尝试...\n")
                        continue
                    else:
                        break
            else:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write("⚠️ 没有新的消息产生\n")
                break
                
        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"❌ 执行出错: {str(e)}\n")
            break
    
    # 在循环结束后，添加运行结束标记
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n🏁 运行结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    # 写入完整轨迹
    trajectory_summary = tracker.get_trajectory_summary()
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n===== AGENT 执行轨迹 =====\n")
        for step in tracker.trajectory:
            f.write(f"步骤 {step['step']:2d} | {step['timestamp']} | {step['agent']:15s} | {step['action']}\n")
            if step['content']:
                f.write(f"       内容: {step['content'][:100]}{'...' if len(step['content']) > 100 else ''}\n")
        f.write(f"\n📊 {trajectory_summary}\n")
        f.write("\n" + "="*80 + "\n\n")

if __name__ == "__main__":
    main()