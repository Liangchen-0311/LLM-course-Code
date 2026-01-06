import json
import re
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    """问题类型枚举"""
    QUERY = "query"           # 查询型问题
    CALCULATION = "calculation"  # 计算型问题
    STATISTICS = "statistics"    # 统计型问题
    GROUP_BY = "group_by"     # 分组查询
    SUBQUERY = "subquery"     # 子查询
    JOIN = "join"             # 多表连接

@dataclass
class ExtractedElements:
    """提取的要素数据类"""
    question_type: QuestionType
    entities: List[str]           # 实体列表
    attributes: List[str]         # 属性列表
    conditions: List[Dict]        # 条件列表
    operations: List[str]         # 操作列表
    time_range: Optional[str]     # 时间范围
    aggregation: Optional[str]    # 聚合函数
    target_metrics: List[str]     # 目标指标
    group_by_fields: List[str]    # 分组字段
    join_tables: List[str]        # 连接表
    subqueries: List[Dict]        # 子查询

class ZhipuElementExtractor:
    """基于智谱AI GLM模型的要素提取器"""
    
    def __init__(self, api_key: str, model: str = "THUDM/glm-4-9b-chat"):
        """
        初始化提取器
        
        Args:
            api_key: 智谱AI API密钥
            model: 使用的模型名称，默认glm-4
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
    
    def _create_prompt(self, question: str) -> str:
        """创建Few-shot Prompt"""
        prompt = f"""你是一个专业的SQL查询分析助手，专门用于分析自然语言问题并提取SQL查询的关键要素。

数据库表结构：
-- 部门表 (dept)
- deptno: 部门编号 (主键)
- dname: 部门名称  
- loc: 部门所在地

-- 员工表 (emp)
- empno: 员工编号 (主键)
- ename: 员工姓名
- job: 员工职位
- mgr: 上级编号 (外键关联emp.empno)
- hiredate: 入职日期
- sal: 员工工资
- comm: 员工奖金
- deptno: 部门编号 (外键关联dept.deptno)

-- 工资等级表 (salgrade)
- grade: 工资等级 (主键)
- losal: 最低工资
- hisal: 最高工资

请分析以下问题并提取SQL查询要素：

问题：{question}

请严格按照以下JSON格式返回结果，不要添加任何其他内容：
{{
    "question_type": "问题类型",
    "entities": ["实体1", "实体2"],
    "attributes": ["属性1", "属性2"],
    "conditions": [
        {{"field": "字段名", "value": "字段值", "operator": "操作符"}}
    ],
    "operations": ["操作1", "操作2"],
    "time_range": "时间范围或null",
    "aggregation": "聚合函数或null",
    "target_metrics": ["目标指标1", "目标指标2"],
    "group_by_fields": ["分组字段1", "分组字段2"],
    "join_tables": ["表1", "表2"],
    "subqueries": [
        {{
            "type": "子查询类型",
            "query": "子查询内容",
            "purpose": "子查询目的"
        }}
    ]
}}

问题类型说明：
- query: 简单查询，如"查询张三的销售业绩"
- calculation: 计算型查询，如"计算所有员工的平均工资"
- statistics: 统计分析，如"分析近3个月的销售趋势"
- group_by: 分组查询，如"显示每个部门的平均工资"
- subquery: 子查询，如"显示工资高于平均工资的员工"
- join: 多表连接，如"显示员工姓名和部门名称"

要素提取规则：
1. entities: 问题中涉及的具体对象（如员工、部门、工资等）
2. attributes: 需要查询或计算的属性（如姓名、工资、部门名称等）
3. conditions: 筛选条件，包含字段名、值和操作符（=, >, <, >=, <=, LIKE, BETWEEN, IN, IS NULL等）
4. operations: 执行的操作类型（如查询、计算、分析、显示、统计等）
5. time_range: 时间限制（如2023年、近3个月等）
6. aggregation: 统计函数（AVG, SUM, COUNT, MAX, MIN等）
7. target_metrics: 最终需要的指标
8. group_by_fields: 分组字段（如部门、职位等）
9. join_tables: 需要连接的表（如emp, dept等）
10. subqueries: 子查询信息

注意事项：
- 字段名应使用数据库中的实际字段名（如ename, sal, deptno等）
- 操作符要准确（如LIKE用于模糊查询，BETWEEN用于范围查询）
- 聚合函数要正确识别（如AVG用于平均值，SUM用于求和）
- 多表连接要识别涉及的表名
- 子查询要识别嵌套查询结构

请确保返回的是有效的JSON格式。"""
        
        return prompt
    
    def _call_zhipu_api(self, prompt: str) -> str:
        """调用智谱AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # 低温度确保输出一致性
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return None
        except KeyError as e:
            print(f"API响应格式错误: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        if not response:
            return {}
        
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            try:
                # 查找JSON开始和结束的位置
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
                    return json.loads(json_str)
            except:
                pass
            
            print(f"无法解析LLM响应: {response}")
            return {}
    
    def _get_question_type_from_string(self, type_str: str) -> QuestionType:
        """从字符串获取问题类型枚举"""
        type_mapping = {
            "query": QuestionType.QUERY,
            "calculation": QuestionType.CALCULATION,
            "statistics": QuestionType.STATISTICS,
            "group_by": QuestionType.GROUP_BY,
            "subquery": QuestionType.SUBQUERY,
            "join": QuestionType.JOIN
        }
        return type_mapping.get(type_str.lower(), QuestionType.QUERY)
    
    def extract_elements(self, question: str) -> ExtractedElements:
        """提取问题要素"""
        # 创建Few-shot prompt
        prompt = self._create_prompt(question)
        
        # 调用智谱AI API
        llm_response = self._call_zhipu_api(prompt)
        
        # 解析响应
        extracted_data = self._parse_llm_response(llm_response)
        
        # 如果API调用失败，使用备用规则
        if not extracted_data:
            extracted_data = self._fallback_extraction(question)
        
        # 从模型响应中获取问题类型
        question_type_str = extracted_data.get("question_type", "query")
        question_type = self._get_question_type_from_string(question_type_str)
        
        return ExtractedElements(
            question_type=question_type,
            entities=extracted_data.get("entities", []),
            attributes=extracted_data.get("attributes", []),
            conditions=extracted_data.get("conditions", []),
            operations=extracted_data.get("operations", []),
            time_range=extracted_data.get("time_range"),
            aggregation=extracted_data.get("aggregation"),
            target_metrics=extracted_data.get("target_metrics", []),
            group_by_fields=extracted_data.get("group_by_fields", []),
            join_tables=extracted_data.get("join_tables", []),
            subqueries=extracted_data.get("subqueries", [])
        )
    
    def _fallback_extraction(self, question: str) -> Dict:
        """备用提取方法（当API调用失败时使用）"""
        # 简化的备用规则，主要用于API失败时的兜底
        question_lower = question.lower()
        
        # 简单的类型判断
        if any(keyword in question_lower for keyword in ["每个", "各组", "分组"]):
            return {"question_type": "group_by"}
        elif any(keyword in question_lower for keyword in ["高于", "低于", "包含"]):
            return {"question_type": "subquery"}
        elif any(keyword in question_lower for keyword in ["连接", "关联", "联合"]):
            return {"question_type": "join"}
        elif any(keyword in question_lower for keyword in ["计算", "统计", "平均", "总和"]):
            return {"question_type": "calculation"}
        elif any(keyword in question_lower for keyword in ["分析", "比较", "趋势"]):
            return {"question_type": "statistics"}
        else:
            return {"question_type": "query"}

def main():
    """主函数 - 演示要素提取"""
    # 请替换为您的智谱AI API密钥
    API_KEY = "sk-"
    
    # 创建提取器
    extractor = ZhipuElementExtractor(api_key=API_KEY)
    
    # 测试问题
    test_questions = [
        "查询张三的销售业绩",
        "计算所有员工的平均工资",
        "分析近3个月的销售趋势",
        "2023年北京地区的销售额是多少？",
        "统计各部门的销售总额",
        "比较各季度业绩表现",
        "我需要分析公司的人力资源情况，包括各部门的员工分布、工资水平统计，以及找出哪些部门的平均工资超过了公司整体平均水平，同时还要考虑员工的职位分布情况，最后生成一份综合性的分析报告",
        "显示工资高于 3000 的员工",
        "查找 1982 年 1 月 1 号以后入职的员工",
        "显示工资在 2000-2500 之间的员工的情况",
        "显示首字母为 s 员工姓名",
        "如何显示第三个字符为大写 o 的所有员工的姓名和工资",
        "如何显示 empno 为 123、345、800 的雇员的情况",
        "显示没有上级的雇员的情况",
        "查询工资高于 500 岗位是 manager 的雇员同时他们的姓名首字母为 J",
        "如何按照工资从高到底来显示",
        "如何按照入职的先后顺序排列",
        "按照部门升序而员工的工资降序排列",
        "统计每个人的年薪并按照从低到高的顺序排列",
        "如何显示工资最高的和工资最低的",
        "如何显示最低工资和该雇员的名字",
        "显示所有员工的品军工资和工资总和",
        "把高于平均工资的雇员的名字和他的工资显示出来",
        "统计共有多少员工",
        "显示每个部门的平均工资和最高工资，并显示部门名称",
        "显示每个部门的每种岗位的平均工资和最低工资",
        "显示平均工资低于 2000 的部门号和它的平均工资",
        "显示雇员的名字，雇员的工资以及所在部门的名称",
        "显示部门名称号为 10 的部门名称、员工和工资",
        "显示雇员名、雇员工资以及所在部门的名字并按部门排序"
    ]
    
    print("=== 基于智谱AI GLM模型的要素提取示例 ===\n")
    
    for question in test_questions:
        print(f"问题：{question}")
        elements = extractor.extract_elements(question)
        
        print(f"问题类型：{elements.question_type.value}")
        print(f"实体：{elements.entities}")
        print(f"属性：{elements.attributes}")
        print(f"条件：{elements.conditions}")
        print(f"操作：{elements.operations}")
        print(f"时间范围：{elements.time_range}")
        print(f"聚合函数：{elements.aggregation}")
        print(f"目标指标：{elements.target_metrics}")
        print(f"分组字段：{elements.group_by_fields}")
        print(f"连接表：{elements.join_tables}")
        print(f"子查询：{elements.subqueries}")
        print("-" * 50)

if __name__ == "__main__":
    main()
    """
 === 基于智谱AI GLM模型的要素提取示例 ===

问题：查询张三的销售业绩
问题类型：query
实体：['张三', '销售业绩']
属性：['员工姓名', '业绩']
条件：[{'field': 'ename', 'value': '张三', 'operator': '='}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['销售业绩']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：计算所有员工的平均工资
问题类型：calculation
实体：['员工']
属性：['工资']
条件：[]
操作：['计算']
时间范围：None
聚合函数：AVG
目标指标：['平均工资']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：分析近3个月的销售趋势
问题类型：statistics
实体：['销售']
属性：['销售趋势']
条件：[{'field': 'hiredate', 'value': '当前日期', 'operator': 'BETWEEN'}, {'field': 'hiredate', 'value': '当前日期 - 3个月', 'operator': 'AND'}]
操作：['分析']
时间范围：近3个月
聚合函数：null
目标指标：['销售趋势']
分组字段：['null']
连接表：['emp', 'dept']
子查询：[{'type': '子查询类型', 'query': 'null', 'purpose': 'null'}]
--------------------------------------------------
问题：2023年北京地区的销售额是多少？
问题类型：calculation
实体：['销售额']
属性：['销售额']
条件：[{'field': 'dept.loc', 'value': '北京', 'operator': '='}]
操作：['计算']
时间范围：null
聚合函数：SUM
目标指标：['销售额']
分组字段：['null']
连接表：['dept', 'emp', 'sales']
子查询：[{'type': '子查询类型', 'query': "SELECT dept.deptno FROM dept WHERE dept.loc = '北京'", 'purpose': '筛选北京地区的部门编号'}]
--------------------------------------------------
问题：统计各部门的销售总额
问题类型：calculation
实体：['部门', '销售总额']
属性：['部门名称', '销售总额']
条件：[{'field': 'dept.deptno', 'value': 'dept.deptno', 'operator': 'IN'}, {'field': 'emp.job', 'value': '销售', 'operator': '='}]
操作：['计算']
时间范围：None
聚合函数：SUM
目标指标：['销售总额']
分组字段：['dept.deptno', 'dept.dname']
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：比较各季度业绩表现
问题类型：calculation
实体：['业绩']
属性：['季度']
条件：[{'field': '业绩', 'value': None, 'operator': '比较'}]
操作：['比较']
时间范围：null
聚合函数：null
目标指标：['季度业绩']
分组字段：['季度']
连接表：['null']
子查询：[{'type': 'null', 'query': 'null', 'purpose': 'null'}]
--------------------------------------------------
问题：我需要分析公司的人力资源情况，包括各部门的员工分布、工资水平统计，以及找出哪些部门的平均工资超过了公司整体平均水平，同时还要考虑员工的职位分布情况，最后生成一份综合性的分析报告
问题类型：statistics
实体：['员工', '部门', '工资', '职位']
属性：['员工姓名', '部门名称', '工资', '奖金', '职位', '部门所在地', '平均工资']
条件：[{'field': 'dept.deptno', 'value': None, 'operator': 'IN'}, {'field': 'dept.loc', 'value': None, 'operator': 'IN'}, {'field': 'salgrade.grade', 'value': None, 'operator': 'IN'}, {'field': 'emp.job', 'value': None, 'operator': 'IN'}]
操作：['分析', '统计', '计算', '显示', '找出']
时间范围：None
聚合函数：AVG
目标指标：['各部门的员工分布', '工资水平统计', '哪些部门的平均工资超过了公司整体平均水平', '员工的职位分布情况']
分组字段：['dept.deptno', 'dept.dname', 'emp.job']
连接表：['dept', 'emp', 'salgrade']
子查询：[{'type': '子查询类型', 'query': 'SELECT AVG(sal) AS avg_salary FROM emp', 'purpose': '计算公司整体平均工资'}, {'type': '子查询类型', 'query': 'SELECT dept.deptno, dept.dname, AVG(emp.sal) AS avg_dept_salary FROM dept JOIN emp ON dept.deptno = emp.deptno GROUP BY dept.deptno, dept.dname', 'purpose': '计算每个部门的平均工资'}]
--------------------------------------------------
问题：显示工资高于 3000 的员工
问题类型：query
实体：['员工']
属性：['工资']
条件：[{'field': 'sal', 'value': '3000', 'operator': '>'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['工资高于3000的员工']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：查找 1982 年 1 月 1 号以后入职的员工
问题类型：query
实体：['员工']
属性：['入职日期']
条件：[{'field': 'hiredate', 'value': '1982-01-01', 'operator': '>='}]
操作：['查询']
时间范围：1982年1月1号以后
聚合函数：None
目标指标：['员工信息']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：显示工资在 2000-2500 之间的员工的情况
问题类型：query
实体：['员工']
属性：['员工编号', '员工姓名', '员工职位', '上级编号', '入职日期', '员工工资', '员工奖金', '部门编号']
条件：[{'field': '员工工资', 'value': '2000', 'operator': '>='}, {'field': '员工工资', 'value': '2500', 'operator': '<='}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['员工编号', '员工姓名', '员工职位', '上级编号', '入职日期', '员工工资', '员工奖金', '部门编号']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：显示首字母为 s 员工姓名
问题类型：query
实体：['员工']
属性：['员工姓名']
条件：[{'field': '员工姓名', 'value': 's%', 'operator': 'LIKE'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['员工姓名']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：如何显示第三个字符为大写 o 的所有员工的姓名和工资
问题类型：query
实体：['员工']
属性：['姓名', '工资']
条件：[{'field': 'ename', 'value': 'o%', 'operator': 'LIKE'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['姓名', '工资']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：如何显示 empno 为 123、345、800 的雇员的情况
问题类型：query
实体：['雇员']
属性：['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
条件：[{'field': 'empno', 'value': '123', 'operator': '='}, {'field': 'empno', 'value': '345', 'operator': '='}, {'field': 'empno', 'value': '800', 'operator': '='}]
操作：['显示']
时间范围：None
聚合函数：None
目标指标：['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：显示没有上级的雇员的情况
问题类型：query
实体：['雇员']
属性：['员工编号', '员工姓名', '员工职位', '上级编号', '入职日期', '员工工资', '员工奖金', '部门编号']
条件：[{'field': 'mgr', 'value': None, 'operator': 'IS NULL'}]
操作：['显示']
时间范围：None
聚合函数：None
目标指标：['员工编号', '员工姓名', '员工职位', '入职日期', '员工工资', '员工奖金', '部门编号']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：查询工资高于 500 岗位是 manager 的雇员同时他们的姓名首字母为 J
问题类型：query
实体：['雇员']
属性：['姓名', '工资', '职位', '上级编号']
条件：[{'field': '工资', 'value': '500', 'operator': '>'}, {'field': '职位', 'value': 'manager', 'operator': '='}, {'field': '姓名', 'value': 'J%', 'operator': 'LIKE'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['姓名']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：如何按照工资从高到底来显示
问题类型：query
实体：['员工']
属性：['工资']
条件：[{'field': 'sal', 'value': None, 'operator': 'DESC'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['工资']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：如何按照入职的先后顺序排列
问题类型：query
实体：['员工']
属性：['入职日期']
条件：[{'field': 'hiredate', 'value': None, 'operator': 'ASC'}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['员工入职顺序']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：按照部门升序而员工的工资降序排列
问题类型：query
实体：['部门', '员工']
属性：['部门名称', '员工工资']
条件：[{'field': 'dept.deptno', 'value': None, 'operator': 'IN'}, {'field': 'emp.deptno', 'value': None, 'operator': '='}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['部门名称', '员工工资']
分组字段：['dept.deptno']
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：统计每个人的年薪并按照从低到高的顺序排列
问题类型：calculation
实体：['员工']
属性：['年薪']
条件：[]
操作：['计算']
时间范围：None
聚合函数：None
目标指标：['年薪']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：如何显示工资最高的和工资最低的
问题类型：query
实体：['员工']
属性：['工资']
条件：[{'field': '工资', 'value': None, 'operator': 'MAX'}, {'field': '工资', 'value': None, 'operator': 'MIN'}]
操作：['查询']
时间范围：None
聚合函数：MAX, MIN
目标指标：['最高工资', '最低工资']
分组字段：[]
连接表：[]
子查询：[]
--------------------------------------------------
问题：如何显示最低工资和该雇员的名字
问题类型：query
实体：['雇员']
属性：['最低工资', '雇员名字']
条件：[{'field': 'sal', 'value': '最低工资', 'operator': '='}, {'field': 'empno', 'value': '该雇员编号', 'operator': '='}]
操作：['查询']
时间范围：None
聚合函数：None
目标指标：['最低工资', '雇员名字']
分组字段：[]
连接表：['emp', 'salgrade']
子查询：[]
--------------------------------------------------
问题：显示所有员工的品军工资和工资总和
问题类型：calculation
实体：['员工']
属性：['品军工资', '工资总和']
条件：[]
操作：['计算']
时间范围：None
聚合函数：SUM
目标指标：['品军工资', '工资总和']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：把高于平均工资的雇员的名字和他的工资显示出来
问题类型：calculation
实体：['雇员']
属性：['名字', '工资']
条件：[{'field': 'sal', 'value': '平均工资', 'operator': '>'}]
操作：['计算', '比较']
时间范围：None
聚合函数：AVG
目标指标：['工资']
分组字段：[]
连接表：['emp']
子查询：[{'type': '子查询类型', 'query': 'SELECT AVG(sal) AS average_salary FROM emp', 'purpose': '计算平均工资'}]
--------------------------------------------------
问题：统计共有多少员工
问题类型：statistics
实体：['员工']
属性：['员工数量']
条件：[]
操作：['统计']
时间范围：null
聚合函数：COUNT
目标指标：['员工总数']
分组字段：[]
连接表：['emp']
子查询：[]
--------------------------------------------------
问题：显示每个部门的平均工资和最高工资，并显示部门名称
问题类型：group_by
实体：['部门']
属性：['部门名称', '平均工资', '最高工资']
条件：[]
操作：['查询']
时间范围：None
聚合函数：AVG(sal), MAX(sal)
目标指标：['平均工资', '最高工资']
分组字段：['部门名称']
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：显示每个部门的每种岗位的平均工资和最低工资
问题类型：group_by
实体：['部门', '岗位']
属性：['部门名称', '岗位', '平均工资', '最低工资']
条件：[{'field': 'dept.deptno', 'value': None, 'operator': '='}, {'field': 'emp.job', 'value': None, 'operator': '='}]
操作：['查询', '计算']
时间范围：None
聚合函数：AVG(sal), MIN(sal)
目标指标：['平均工资', '最低工资']
分组字段：['dept.dname', 'emp.job']
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：显示平均工资低于 2000 的部门号和它的平均工资
问题类型：calculation
实体：['部门', '工资']
属性：['部门号', '平均工资']
条件：[{'field': '平均工资', 'value': '2000', 'operator': '<'}]
操作：['计算', '显示']
时间范围：None
聚合函数：AVG
目标指标：['部门号', '平均工资']
分组字段：['部门号']
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：显示雇员的名字，雇员的工资以及所在部门的名称
问题类型：join
实体：['雇员', '部门']
属性：['雇员的名字', '雇员的工资', '所在部门的名称']
条件：[{'field': 'emp.deptno', 'value': 'dept.deptno', 'operator': '='}]
操作：['显示']
时间范围：None
聚合函数：None
目标指标：['雇员的名字', '雇员的工资', '所在部门的名称']
分组字段：[]
连接表：['emp', 'dept']
子查询：[]
--------------------------------------------------
问题：显示部门名称号为 10 的部门名称、员工和工资
问题类型：join
实体：['部门', '员工', '工资']
属性：['部门名称', '员工姓名', '工资']
条件：[{'field': 'dept.deptno', 'value': '10', 'operator': '='}]
操作：['显示']
时间范围：None
聚合函数：None
目标指标：['部门名称', '员工姓名', '工资']
分组字段：[]
连接表：['dept', 'emp']
子查询：[]
--------------------------------------------------
问题：显示雇员名、雇员工资以及所在部门的名字并按部门排序
问题类型：join
实体：['雇员', '部门']
属性：['雇员名', '雇员工资', '部门名字']
条件：[{'field': 'deptno', 'value': 'dept.deptno', 'operator': '='}, {'field': 'deptno', 'value': 'emp.deptno', 'operator': '='}]
操作：['显示']
时间范围：None
聚合函数：None
目标指标：['雇员名', '雇员工资', '部门名字']
分组字段：['部门名字']
连接表：['emp', 'dept']
子查询：[]
--------------------------------------------------
    
    """