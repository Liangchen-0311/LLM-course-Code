import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from dataclasses import dataclass
import json
import re
from transformers import AutoTokenizer, AutoModel
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 导入意图识别模块
from intent_recognition_feature_extraction import ZhipuElementExtractor, ExtractedElements

@dataclass
class DatabaseField:
    """数据库字段信息"""
    table_name: str
    field_name: str
    field_type: str
    description: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None

@dataclass
class IntentElement:
    """意图要素信息"""
    element_type: str  # entities, attributes, conditions, operations, target_metrics
    element_value: str
    confidence: float = 0.0

class BGE_FieldMatcher:
    """使用BGE模型进行意图要素和数据库字段相似度匹配"""
    
    def __init__(self):
        """
        初始化BGE字段匹配器
        
        Args:
            model_name: BGE模型名称
            cache_folder: 模型缓存文件夹
        """
        local_model_path = r"D:/developer/vscode_workspace/llm/models/bge-small-zh-v1.5"
        # 直接使用 transformers 库
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.database_fields = self._initialize_database_fields()
        self.field_embeddings = None
        self._compute_field_embeddings()
    
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """计算文本的嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 对文本进行编码
        encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                max_length=512, return_tensors='pt')
        
        # 移动到设备
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 计算嵌入
        with torch.no_grad():
            outputs = self.model(**encoded)
            # 使用 [CLS] token 的输出作为句子嵌入
            embeddings = outputs.last_hidden_state[:, 0, :]
            # 归一化嵌入向量
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _initialize_database_fields(self) -> List[DatabaseField]:
        """初始化数据库字段信息"""
        fields = [
            # 部门表 (dept)
            DatabaseField("dept", "deptno", "INT", "部门编号", is_primary_key=True),
            DatabaseField("dept", "dname", "VARCHAR", "部门名称"),
            DatabaseField("dept", "loc", "VARCHAR", "部门所在地"),
            
            # 员工表 (emp)
            DatabaseField("emp", "empno", "INT", "员工编号", is_primary_key=True),
            DatabaseField("emp", "ename", "VARCHAR", "员工姓名"),
            DatabaseField("emp", "job", "VARCHAR", "员工职位"),
            DatabaseField("emp", "mgr", "INT", "上级编号", is_foreign_key=True, referenced_table="emp"),
            DatabaseField("emp", "hiredate", "DATE", "入职日期"),
            DatabaseField("emp", "sal", "DECIMAL", "员工工资"),
            DatabaseField("emp", "comm", "DECIMAL", "员工奖金"),
            DatabaseField("emp", "deptno", "INT", "部门编号", is_foreign_key=True, referenced_table="dept"),
            
            # 工资等级表 (salgrade)
            DatabaseField("salgrade", "grade", "INT", "工资等级", is_primary_key=True),
            DatabaseField("salgrade", "losal", "DECIMAL", "最低工资"),
            DatabaseField("salgrade", "hisal", "DECIMAL", "最高工资"),
        ]
        return fields
    
    def _compute_field_embeddings(self):
        """计算所有数据库字段的嵌入向量"""
        field_texts = []
        for field in self.database_fields:
            # 构建字段的文本表示
            field_text = f"{field.table_name}.{field.field_name} {field.description}"
            if field.is_primary_key:
                field_text += " 主键"
            if field.is_foreign_key:
                field_text += f" 外键关联{field.referenced_table}"
            field_texts.append(field_text)
        
        """
        field_text = ['dept.deptno 部门编号 主键', 
                        'dept.dname 部门名称', 
                        'dept.loc 部门所在地',
                        'emp.empno 员工编号 主键', 
                        'emp.ename 员工姓名', 
                        'emp.job 员工职位',
                        'emp.mgr 上级编号 外键关联emp',
                        'emp.hiredate 入职日期', 
                        'emp.sal 员工工资', 
                        'emp.comm 员工奖金',
                        'emp.deptno 部门编号 外键关联dept', 
                        'salgrade.grade 工资等级 主键', 
                        'salgrade.losal 最低工资',
                        'salgrade.hisal 最高工资']
        """
        # 使用BGE模型计算嵌入向量
        self.field_embeddings = self._encode_text(field_texts)
    
    def _normalize_intent_element(self, element: str) -> str:
        """标准化意图要素文本"""
        # 移除特殊字符，保留中文和英文
        normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', element)
        # 移除多余空格
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def compute_similarity(self, intent_element: str, field_embedding: np.ndarray) -> float:
        """计算意图要素与字段嵌入向量的相似度"""
        # 标准化意图要素
        normalized_element = self._normalize_intent_element(intent_element)
        
        # 计算意图要素的嵌入向量
        element_embedding = self._encode_text(normalized_element)
        element_embedding = element_embedding.flatten()  # 确保是一维数组
        
        # 计算余弦相似度
        similarity = np.dot(element_embedding, field_embedding)
        return float(similarity)
    
    def match_intent_elements(self, intent_elements: List[IntentElement], 
                            top_k: int = 3, 
                            similarity_threshold: float = 0.3) -> Dict[str, List[Tuple[DatabaseField, float]]]:
        """
        匹配意图要素与数据库字段
        
        Args:
            intent_elements: 意图要素列表
            top_k: 返回前k个最相似的字段
            similarity_threshold: 相似度阈值
            
        Returns:
            匹配结果字典，键为要素类型，值为(字段, 相似度)元组列表
        
        这里是把每一个意图要素(这里要把意图展平了看，) 和 库里面的三张表的每一个字段计算下相似度
        """
        matches = {}
        
        for element in intent_elements:
            element_type = element.element_type
            element_value = element.element_value
            
            # 计算与所有字段的相似度
            similarities = []
            for i, field in enumerate(self.database_fields):
                similarity = self.compute_similarity(element_value, self.field_embeddings[i])
                if similarity >= similarity_threshold:
                    similarities.append((field, similarity))
            
            # 按相似度排序，取前top_k个
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:top_k]
            
            if top_matches:
                matches[element_type] = top_matches
        
        return matches
    
    def get_field_by_name(self, table_name: str, field_name: str) -> Optional[DatabaseField]:
        """根据表名和字段名获取字段信息"""
        for field in self.database_fields:
            if field.table_name == table_name and field.field_name == field_name:
                return field
        return None
    
    def get_fields_by_table(self, table_name: str) -> List[DatabaseField]:
        """获取指定表的所有字段"""
        return [field for field in self.database_fields if field.table_name == table_name]
    
    def get_all_tables(self) -> List[str]:
        """获取所有表名"""
        return list(set(field.table_name for field in self.database_fields))

def create_intent_elements_from_extracted_elements(elements: ExtractedElements) -> List[IntentElement]:
    """从ExtractedElements对象创建意图要素列表"""
    intent_elements = []
    
    # 处理实体
    for entity in elements.entities:
        intent_elements.append(IntentElement("entities", entity))
    
    # 处理属性
    for attribute in elements.attributes:
        intent_elements.append(IntentElement("attributes", attribute))
    
    # 处理条件中的字段
    for condition in elements.conditions:
        field = condition.get("field", "")
        if field:
            intent_elements.append(IntentElement("conditions", field))
    
    # 处理操作
    for operation in elements.operations:
        intent_elements.append(IntentElement("operations", operation))
    
    # 处理目标指标
    for metric in elements.target_metrics:
        intent_elements.append(IntentElement("target_metrics", metric))
    
    # 处理分组字段
    for group_field in elements.group_by_fields:
        intent_elements.append(IntentElement("group_by_fields", group_field))
    
    return intent_elements

def main():
    """主函数 - 演示字段匹配功能"""
    # 初始化BGE字段匹配器
    matcher = BGE_FieldMatcher()
    
    print("=== BGE 数据库字段匹配器 ===")
    print(f"数据库表数量: {len(matcher.get_all_tables())}")
    print(f"字段总数: {len(matcher.database_fields)}")
    print()
    
    # 示例2: 匹配从意图识别中提取的要素
    print("示例2: 匹配意图识别要素")
    
    # 使用意图识别模块提取要素
    API_KEY = "sk-"  # 请替换为您的API密钥
    extractor = ZhipuElementExtractor(api_key=API_KEY)
    
    # 测试问题列表
    test_questions = [
        # "查询张三的销售业绩",
        # "计算所有员工的平均工资",
        # "分析近3个月的销售趋势",
        # "2023年北京地区的销售额是多少？",
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
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试问题 {i}: {question} ---")
        
        # 提取要素
        elements = extractor.extract_elements(question)
        
        # 打印提取的意图要素
        print(f"\n提取的意图要素:")
        print(f"  问题类型: {elements.question_type.value}")
        print(f"  实体: {elements.entities}")
        print(f"  属性: {elements.attributes}")
        print(f"  条件: {elements.conditions}")
        print(f"  操作: {elements.operations}")
        print(f"  时间范围: {elements.time_range}")
        print(f"  聚合函数: {elements.aggregation}")
        print(f"  目标指标: {elements.target_metrics}")
        print(f"  分组字段: {elements.group_by_fields}")
        print(f"  连接表: {elements.join_tables}")
        print(f"  子查询: {elements.subqueries}")
        
        # 创建意图要素
        intent_elements = create_intent_elements_from_extracted_elements(elements)
        
        # 进行匹配
        matches = matcher.match_intent_elements(intent_elements, top_k=2, similarity_threshold=0.2)
        
        # 显示匹配结果
        print(f"\n字段匹配结果:")
        for element_type, field_matches in matches.items():
            print(f"\n  {element_type}:")
            for field, similarity in field_matches:
                print(f"    - {field.table_name}.{field.field_name} ({field.description}): {similarity:.3f}")
        
        print("\n" + "-" * 80)
    
    print("\n" + "="*50)
    
    # 示例3: 显示数据库结构
    print("示例3: 数据库结构")
    for table_name in matcher.get_all_tables():
        fields = matcher.get_fields_by_table(table_name)
        print(f"\n表: {table_name}")
        for field in fields:
            pk_fk_info = ""
            if field.is_primary_key:
                pk_fk_info = " (主键)"
            elif field.is_foreign_key:
                pk_fk_info = f" (外键 -> {field.referenced_table})"
            print(f"  - {field.field_name}: {field.description}{pk_fk_info}")

if __name__ == "__main__":
    main()

"""
--- 测试问题 29: 显示部门名称号为 10 的部门名称、员工和工资 ---

提取的意图要素:
  问题类型: join
  实体: ['部门', '员工', '工资']
  属性: ['部门名称', '员工姓名', '工资']
  条件: [{'field': 'dept.deptno', 'value': '10', 'operator': '='}]
  操作: ['显示']
  时间范围: None
  聚合函数: None
  目标指标: ['部门名称', '员工姓名', '工资']
  分组字段: []
  连接表: ['dept', 'emp']
  子查询: []

字段匹配结果:

  entities:
    - salgrade.grade (工资等级): 0.535
    - emp.sal (员工工资): 0.533

  attributes:
    - salgrade.grade (工资等级): 0.535
    - emp.sal (员工工资): 0.533

  conditions:
    - dept.deptno (部门编号): 0.644
    - emp.deptno (部门编号): 0.614

  operations:
    - salgrade.grade (工资等级): 0.320
    - emp.mgr (上级编号): 0.294

  target_metrics:
    - salgrade.grade (工资等级): 0.535
    - emp.sal (员工工资): 0.533

--------------------------------------------------------------------------------

--- 测试问题 30: 显示雇员名、雇员工资以及所在部门的名字并按部门排序 ---

提取的意图要素:
  问题类型: join
  实体: ['雇员', '部门']
  属性: ['雇员名', '雇员工资', '部门名称']
  条件: [{'field': 'deptno', 'value': 'dept.deptno', 'operator': '='}, {'field': 'deptno', 'value': 'dept.deptno', 'operator': '='}]
  操作: ['显示', '连接']
  时间范围: None
  聚合函数: None
  目标指标: ['雇员名', '雇员工资', '部门名称']
  分组字段: ['部门名称']
  连接表: ['emp', 'dept']
  子查询: []

字段匹配结果:

  entities:
    - dept.dname (部门名称): 0.484
    - dept.loc (部门所在地): 0.479

  attributes:
    - dept.dname (部门名称): 0.587
    - dept.loc (部门所在地): 0.503

  conditions:
    - dept.deptno (部门编号): 0.607
    - emp.deptno (部门编号): 0.588

  operations:
    - emp.deptno (部门编号): 0.306
    - emp.mgr (上级编号): 0.292

  target_metrics:
    - dept.dname (部门名称): 0.587
    - dept.loc (部门所在地): 0.503

  group_by_fields:
    - dept.dname (部门名称): 0.587
    - dept.loc (部门所在地): 0.503

"""