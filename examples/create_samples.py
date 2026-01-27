"""
示例脚本 - 创建示例Excel文件用于测试
Example script to create sample Excel files for testing.
"""

import pandas as pd
from pathlib import Path

# 创建示例数据目录
examples_dir = Path(__file__).parent
examples_dir.mkdir(exist_ok=True)

# 示例1: 财务报表
def create_financial_report():
    """创建财务报表示例"""
    # 收入表
    income_data = {
        '月份': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'],
        '营业收入': [1500000, 1650000, 1800000, 1750000, 1900000, 2100000],
        '营业成本': [900000, 990000, 1080000, 1050000, 1140000, 1260000],
        '毛利润': [600000, 660000, 720000, 700000, 760000, 840000],
        '销售费用': [150000, 165000, 180000, 175000, 190000, 210000],
        '管理费用': [120000, 132000, 144000, 140000, 152000, 168000],
        '净利润': [330000, 363000, 396000, 385000, 418000, 462000]
    }
    income_df = pd.DataFrame(income_data)
    
    # 资产负债表
    balance_data = {
        '科目': ['现金', '应收账款', '存货', '固定资产', '总资产', '应付账款', '短期借款', '长期借款', '总负债', '净资产'],
        '2024-Q1': [500000, 800000, 600000, 2000000, 3900000, 400000, 300000, 1000000, 1700000, 2200000],
        '2024-Q2': [550000, 900000, 650000, 2100000, 4200000, 450000, 350000, 1000000, 1800000, 2400000]
    }
    balance_df = pd.DataFrame(balance_data)
    
    # 保存到Excel
    output_path = examples_dir / 'financial_report.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        income_df.to_excel(writer, sheet_name='损益表', index=False)
        balance_df.to_excel(writer, sheet_name='资产负债表', index=False)
    
    print(f"创建示例文件: {output_path}")


# 示例2: 销售数据
def create_sales_data():
    """创建销售数据示例"""
    # 销售明细
    sales_detail = {
        '日期': pd.date_range('2024-01-01', periods=30, freq='D'),
        '产品': ['产品A', '产品B', '产品C'] * 10,
        '销售数量': [100, 150, 80, 120, 200, 90, 110, 180, 95, 130, 
                    140, 85, 105, 190, 100, 125, 160, 88, 115, 175,
                    92, 135, 145, 87, 108, 195, 102, 128, 165, 91],
        '单价': [50, 60, 45, 50, 60, 45, 50, 60, 45, 50,
                60, 45, 50, 60, 45, 50, 60, 45, 50, 60,
                45, 50, 60, 45, 50, 60, 45, 50, 60, 45],
        '销售额': None,
        '区域': ['华北', '华东', '华南'] * 10
    }
    sales_df = pd.DataFrame(sales_detail)
    sales_df['销售额'] = sales_df['销售数量'] * sales_df['单价']
    
    # 产品信息
    product_info = {
        '产品名称': ['产品A', '产品B', '产品C'],
        '类别': ['电子产品', '电子产品', '家居用品'],
        '成本': [30, 40, 25],
        '库存': [500, 800, 600]
    }
    product_df = pd.DataFrame(product_info)
    
    # 保存到Excel
    output_path = examples_dir / 'sales_data.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        sales_df.to_excel(writer, sheet_name='销售明细', index=False)
        product_df.to_excel(writer, sheet_name='产品信息', index=False)
    
    print(f"创建示例文件: {output_path}")


# 示例3: 员工薪资
def create_employee_salary():
    """创建员工薪资示例"""
    salary_data = {
        '员工编号': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008'],
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
        '部门': ['财务部', '销售部', '技术部', '财务部', '销售部', '技术部', '人力资源部', '技术部'],
        '职位': ['会计', '销售经理', '工程师', '出纳', '销售代表', '工程师', 'HR专员', '工程师'],
        '基本工资': [8000, 12000, 15000, 6000, 8000, 15000, 7000, 14000],
        '绩效奖金': [2000, 5000, 3000, 1000, 3000, 3500, 1500, 3200],
        '补贴': [500, 800, 600, 400, 500, 600, 400, 600],
        '应发工资': None
    }
    salary_df = pd.DataFrame(salary_data)
    salary_df['应发工资'] = salary_df['基本工资'] + salary_df['绩效奖金'] + salary_df['补贴']
    
    output_path = examples_dir / 'employee_salary.xlsx'
    salary_df.to_excel(output_path, index=False, sheet_name='薪资表')
    
    print(f"创建示例文件: {output_path}")


if __name__ == '__main__':
    create_financial_report()
    create_sales_data()
    create_employee_salary()
    print("\n所有示例文件创建完成！")
