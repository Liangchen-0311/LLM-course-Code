from intent_recognition_feature_extraction import ZhipuElementExtractor
import json
from db_field_recall import BGE_FieldMatcher, create_intent_elements_from_extracted_elements

"""
百度给出的  训练数据格式如下：
{
  "instruction": "生成查询2025年1月订单总额的SQL",
  "input": "表名:orders, 金额字段:amount, 时间字段:create_time",
  "output": "SELECT SUM(amount) AS total FROM orders WHERE create_time BETWEEN '2025-01-01' AND '2025-01-31'"
}

# 生成SQL示例
input_text = "生成查询2025年8月订单量的SQL，表名orders，时间字段create_time"
response, _ = model.chat(tokenizer, input_text, history=[])
print(response)


KIMI给出的训练样例:
[
  {
    "instruction": "查询用户表中用户ID为1的用户的姓名和年龄。",
    "sql_template": "SELECT column1, column2 FROM table_name WHERE condition;",
    "real_sql": "SELECT name, age FROM users WHERE user_id = 1;"
  },
  {
    "instruction": "计算每个部门的员工平均工资。",
    "sql_template": "SELECT column1, AGG_FUNC(column2) AS calculated_value FROM table_name GROUP BY column1;",
    "real_sql": "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department;"
  },
  {
    "instruction": "统计每个部门中工资超过5000的员工数量和平均工资。",
    "sql_template": "SELECT column1, COUNT(*) AS count_value, AGG_FUNC(column2) AS calculated_value FROM table_name WHERE condition GROUP BY column1 HAVING condition2;",
    "real_sql": "SELECT department, COUNT(*) AS employee_count, AVG(salary) AS avg_salary FROM employees WHERE salary > 5000 GROUP BY department;"
  },
  {
    "instruction": "查询订单表中订单金额大于1000的订单编号和订单日期。",
    "sql_template": "SELECT column1, column2 FROM table_name WHERE condition;",
    "real_sql": "SELECT order_id, order_date FROM orders WHERE order_amount > 1000;"
  },
  {
    "instruction": "计算每个产品的总销售额。",
    "sql_template": "SELECT column1, AGG_FUNC(column2) AS calculated_value FROM table_name GROUP BY column1;",
    "real_sql": "SELECT product_id, SUM(sales_amount) AS total_sales FROM sales GROUP BY product_id;"
  }
]
####################################################
request_data = {
  "instruction": "查询用户表中用户ID为1的用户的姓名和年龄。",
  "sql_template": "SELECT column1, column2 FROM table_name WHERE condition;"
}

# 发送POST请求
response = requests.post(url, json=request_data)

"""

def generate_sql_templates_by_zhipu(api_key):
    extractor = ZhipuElementExtractor(api_key=api_key)
    prompt = """
你是一个SQL专家，请为以下六种常见的SQL查询类型，分别给出标准的SQL模板（用大括号占位符），每种类型只给出模板，不要举例，不要多余解释，返回JSON格式：

{
  "query": "SELECT {select_fields} FROM {main_table} {where_clause}",
  "calculation": "SELECT {aggregation}({select_fields}) FROM {main_table} {where_clause}",
  "statistics": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
  "group_by": "SELECT {group_by_fields}, {aggregation}({select_fields}) FROM {main_table} {where_clause} GROUP BY {group_by_fields}",
  "subquery": "SELECT {select_fields} FROM {main_table} WHERE {field} {operator} (SELECT {sub_select_fields} FROM {sub_table} {sub_where_clause})",
  "join": "SELECT {select_fields} FROM {main_table} JOIN {join_table} ON {join_condition} {where_clause}"
}
"""
    # 调用大模型
    response = extractor._call_zhipu_api(prompt)
    # 解析JSON
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end]
        sql_templates = json.loads(json_str)
        return sql_templates
    except Exception as e:
        print("解析大模型返回的SQL模板失败:", e)
        return None

def get_best_field_name(field, field_matches):
    # 在 field_matches 里查找与 field 最相关的真实字段名
    for t in field_matches:
        for f, _ in field_matches[t]:
            if field in f.description or field in f.field_name:
                return f"{f.table_name}.{f.field_name}"
    # fallback
    return field

def get_best_fields(types, main_table, field_matches, top_k=2, allow_multi_table=False):
    fields = []
    seen = set()
    for t in types:
        if t in field_matches:
            for field, _ in field_matches[t][:top_k]:
                field_str = f"{field.table_name}.{field.field_name}"
                # 只保留主表字段，除非是 join 查询
                if allow_multi_table or field.table_name == main_table:
                    if field_str not in seen:
                        fields.append(field_str)
                        seen.add(field_str)
    return ", ".join(fields) if fields else "*"

def fill_sql_template(elements, field_matches, sql_templates):
    qtype = elements.question_type.value
    template = sql_templates.get(qtype, sql_templates.get("query"))
    
    # 选字段
    main_table = elements.join_tables[0] if elements.join_tables else "emp"
    allow_multi_table = elements.question_type.value == "join"
    select_fields = get_best_fields(["attributes", "target_metrics", "entities"], main_table, field_matches, top_k=2, allow_multi_table=allow_multi_table)
    
    where_clause = ""
    if elements.conditions:
        conds = []
        for cond in elements.conditions:
            field = cond.get("field", "")
            op = cond.get("operator", "=")
            value = cond.get("value", "")
            # 这里做字段名映射
            real_field = get_best_field_name(field, field_matches)
            if op.upper() == "LIKE":
                conds.append(f"{real_field} LIKE '{value}'")
            elif op.upper() == "IS NULL":
                conds.append(f"{real_field} IS NULL")
            elif op.upper() == "IN":
                conds.append(f"{real_field} IN ({value})")
            elif op.upper() == "BETWEEN" and isinstance(value, list) and len(value) == 2:
                conds.append(f"{real_field} BETWEEN {value[0]} AND {value[1]}")
            else:
                conds.append(f"{real_field} {op} {repr(value)}")
        where_clause = "WHERE " + " AND ".join(conds)
    aggregation = elements.aggregation or "SUM"
    group_by_fields = get_best_fields(["group_by_fields"], main_table, field_matches, allow_multi_table=allow_multi_table)
    join_table = elements.join_tables[1] if len(elements.join_tables) > 1 else ""
    join_condition = ""
    if elements.conditions:
        for cond in elements.conditions:
            field = cond.get("field", "")
            value = cond.get("value", "")
            if isinstance(field, str) and isinstance(value, str) and "." in field and "." in value:
                join_condition = f"{field} = {value}"
                break
    if not join_condition and join_table:
        join_condition = f"{main_table}.deptno = {join_table}.deptno"
    # subquery
    sub_select_fields = select_fields
    sub_table = main_table
    sub_where_clause = where_clause
    field = select_fields.split(",")[0] if select_fields else "*"
    operator = "="
    
    # 填充模板
    sql = template.format(
        select_fields=select_fields,
        main_table=main_table,
        where_clause=where_clause,
        aggregation=aggregation,
        group_by_fields=group_by_fields,
        join_table=join_table,
        join_condition=join_condition,
        sub_select_fields=sub_select_fields,
        sub_table=sub_table,
        sub_where_clause=sub_where_clause,
        field=field,
        operator=operator
    )
    # 清理多余空格
    sql = " ".join(sql.split())
    return sql

# 用法示例
if __name__ == "__main__":
    API_KEY = "sk-"
    templates = generate_sql_templates_by_zhipu(API_KEY)
    print("SQL_TEMPLATES:", templates)
    
    # 初始化意图识别和字段召回
    extractor = ZhipuElementExtractor(api_key=API_KEY)
    matcher = BGE_FieldMatcher()
    
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
    
    for idx, question in enumerate(test_questions, 1):
        print(f"\n【问题{idx}】{question}")
        # 1. 要素提取
        elements = extractor.extract_elements(question)
        # 2. 字段召回
        intent_elements = create_intent_elements_from_extracted_elements(elements)
        field_matches = matcher.match_intent_elements(intent_elements, top_k=2, similarity_threshold=0.2)
        # 3. SQL生成
        sql = fill_sql_template(elements, field_matches, templates)
        print("生成SQL：", sql)
