import requests
import json
import pandas as pd
import os

# ================= 配置区域 =================

# 1. 您的 Access Token (保持不变)
TAT = ""

# 2. 飞书表格 Token (您确认有效的这个)
SPREADSHEET_TOKEN = ""

# 3. 要上传的文件名
CSV_FILE = "result.csv"

# ===========================================

def get_header():
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + TAT
    }

def get_first_sheet_id():
    """
    使用 V3 接口获取表格中的第一个工作表 ID
    对应您提供的官方 SDK: client.sheets.v3.spreadsheet_sheet.query
    """
    url = f"https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/{SPREADSHEET_TOKEN}/sheets/query"
    
    try:
        r = requests.get(url, headers=get_header())
        resp = r.json()
        
        if resp.get("code") == 0:
            # 获取第一个工作表的 sheet_id
            sheets = resp["data"]["sheets"]
            if sheets:
                first_sheet = sheets[0]
                sheet_id = first_sheet["sheet_id"]
                title = first_sheet["title"]
                print(f"✅ 成功连接表格！")
                print(f"   找到工作表: '{title}' (ID: {sheet_id})")
                return sheet_id
            else:
                print("❌ 表格中没有找到任何工作表 (sheet)。")
                return None
        else:
            print(f"❌ 获取工作表失败: {resp.get('msg')}")
            print(f"   错误码: {resp.get('code')}")
            return None
            
    except Exception as e:
        print(f"❌ 网络请求出错: {e}")
        return None

def get_excel_column_name(n):
    """数字转Excel列名 (1->A, 27->AA)"""
    name = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        name = chr(r + ord('A')) + name
    return name

def upload_data(sheet_id):
    """读取CSV并上传数据"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到文件 {CSV_FILE}")
        return

    # 读取 CSV
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.fillna("") # 处理空值
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    header = df.columns.tolist()
    data_rows = df.values.tolist()
    all_values = [header] + data_rows
    
    # 计算范围
    row_count = len(all_values)
    col_count = len(header)
    end_col = get_excel_column_name(col_count)
    
    # 构造 V2 写入接口 (V2 接口写入数据最稳定)
    url = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{SPREADSHEET_TOKEN}/values"
    
    range_str = f"{sheet_id}!A1:{end_col}{row_count}"
    
    post_data = {
        "valueRange": {
            "range": range_str,
            "values": all_values
        }
    }
    
    print(f"🚀 开始上传数据到范围: {range_str} ...")
    
    try:
        r = requests.put(url, headers=get_header(), data=json.dumps(post_data))
        resp = r.json()
        
        if resp.get("code") == 0:
            print(f"✅ 上传成功！")
            print(f"   更新了 {resp['data']['revision']} 个单元格。")
            print(f"   请去飞书查看结果。")
        else:
            print(f"❌ 上传失败: {resp}")
            
    except Exception as e:
        print(f"❌ 上传请求出错: {e}")

if __name__ == "__main__":
    print("--- 开始执行 ---")
    # 1. 自动获取 sheet_id
    sheet_id = get_first_sheet_id()
    
    # 2. 如果获取成功，执行上传
    if sheet_id:
        upload_data(sheet_id)