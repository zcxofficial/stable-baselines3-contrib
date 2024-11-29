import pandas as pd
import numpy as np
import os
import re
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import pearsonr

''' Operatiors Defiantion'''

# Unary Operators
def Abs(df):
    return df.abs()

def Sign(df):
    return np.sign(df)

def Log(df):
    return np.log(df)

def CSRank(df):
    return df.rank(axis=1, pct=True)

# Binary Operators
def Add(df1, df2):
    return df1 + df2

def Sub(df1, df2):
    return df1 - df2

def Mul(df1, df2):
    return df1 * df2

def Div(df1, df2):
    return df1 / df2

def Pow(df1, df2):
    return df1 ** df2

def Greater(df1, df2):
    # Returns the greater of df1 and df2 element-wise
    return np.maximum(df1, df2)

def Less(df1, df2):
    # Returns the lesser of df1 and df2 element-wise
    return np.minimum(df1, df2)

# Rolling Operators
def Ref(df, periods):
    return df.shift(periods)

def Mean(df, window):
    return df.rolling(window=window).mean()

def Sum(df, window):
    return df.rolling(window=window).sum()

def Std(df, window):
    return df.rolling(window=window).std()

def Var(df, window):
    return df.rolling(window=window).var()

def Skew(df, window):
    return df.rolling(window=window).skew()

def Kurt(df, window):
    return df.rolling(window=window).kurt()

def Max(df, window):
    return df.rolling(window=window).max()

def Min(df, window):
    return df.rolling(window=window).min()

def Med(df, window):
    return df.rolling(window=window).median()

def Mad(df, window):
    return df.rolling(window=window).apply(lambda x: np.fabs(x - np.median(x)).mean(), raw=True)

def Rank(df, window):
    return df.rolling(window=window).apply(lambda x: x.rank(pct=True)[-1], raw=False)

def Delta(df, periods):
    return df.diff(periods)

def WMA(df, window):
    weights = np.arange(1, window + 1)
    return df.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def EMA(df, span):
    return df.ewm(span=span, adjust=False).mean()

# Pair Rolling Operators
def Cov(df1, df2, window):
    return df1.rolling(window=window).cov(df2)

def Corr(df1, df2, window):
    return df1.rolling(window=window).corr(df2)

def to_percentage(y, _):
    return f"{y:.1f}%"

def quarter_formatter(x, pos=None):
    date = pd.to_datetime(x)
    if date.is_quarter_start:
        return date.strftime('%Y-%m-%d')
    else:
        return ""

class Backtest(object):

    def __init__(self,
                 ohlc_data_path: str = "",
                 rl_exprs_path: str = "",
                 backtest_time: list = [],
                 max_workers: int = 10,
            ):

        self._ohlc_data_path = os.path.expanduser(ohlc_data_path)
        self._rl_exprs_path = os.path.expanduser(rl_exprs_path)
        self._backtest_start_time = backtest_time[0]
        self._backtest_end_time = backtest_time[1]
        '''read daily stock data'''
        features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        olhc_data_dict = {feature: [] for feature in features}
        # 遍历目录中的所有CSV文件
        for filename in os.listdir(self._ohlc_data_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(self._ohlc_data_path, filename)
                df_full = pd.read_csv(filepath)
                stock_code = filename.replace('.csv', '')
                for feature in features:
                    if feature in df_full.columns:
                        df = df_full[['date', feature]].copy()
                        df.set_index('date', inplace=True)
                        df.rename(columns={feature: stock_code}, inplace=True)
                        olhc_data_dict[feature].append(df)

        for feature in features:
            combined_df = pd.concat(olhc_data_dict[feature], axis=1)
            olhc_data_dict[feature] = combined_df

        self._olhc_data_dict = olhc_data_dict

    # 分组计算
    def quantile_calc(self, x, _quantiles):
        """

        :param x: 因子值
        :param _quantiles: 分组数量
        :return:
        """
        quantiles = pd.qcut(x, _quantiles, labels=False, duplicates='drop') + 1

        # return quantiles.sort_index()
        return quantiles

    # 分组回测计算
    def factor_group_backtest_cal(self, factor_df, ret_name, factor_name, group_num_backtest=5):
        factor_df_copy = factor_df.dropna(
            subset=[factor_name]).copy()  # 全局变量

        # 每天每批次分组
        factor_df_copy['group_backtest_date'] = factor_df_copy.groupby(['dateStr'], as_index=False, group_keys=False)[
            factor_name].apply(
            self.quantile_calc,
            group_num_backtest)

        # 每天分组因子值分布
        group_date_factor_mean = factor_df_copy.groupby(
            ['group_backtest_date'])[factor_name].mean()  # 中间值

        # 每天分组收益率分布
        group_date_ret_mean = factor_df_copy.groupby(['group_backtest_date'])[
            ret_name].mean()  # 中间值

        # 每天分组累积收益率
        group_date_ret_cumsum = factor_df_copy.groupby(['dateStr', 'group_backtest_date'])[
            ret_name].mean().unstack(
            level=1).cumsum()  # 中间值

        return group_date_factor_mean, group_date_ret_mean, group_date_ret_cumsum

    def calc_factor_value(self, expressions):
        for feature_name, dataframe in self._olhc_data_dict.items():
            exec(f"{feature_name} = self._olhc_data_dict['{feature_name}']")
        factor_value_list = []
        for index, expression in enumerate(expressions, start=1):
            variable_name = f"rlfactor_{index}"
            exec_code = f"{variable_name} = {expression}"
            try:
                exec(exec_code)
            except:
                print(f"Warning: {exec_code}")
            exec(
                f"rlfactor_{index} = pd.melt(rlfactor_{index}.reset_index(), id_vars=['date'], var_name='code', value_name='value')")
            exec(
                f"rlfactor_{index} = rlfactor_{index}[(rlfactor_{index}['date'] >= self._backtest_start_time) & (rlfactor_{index}['date'] <= self._backtest_end_time)]")
            exec(f"factor_value_list.append(rlfactor_{index})")

        return factor_value_list

def infix_to_postfix(expression):

    ''' 把标准数学表达式转化为RPN序列表示'''

    precedence = {
        'Abs': 4,  # 单目运算符，较高优先级
        'Log': 4,  # 单目运算符，较高优先级
        'Add': 2,  # 双目算术运算符，中等优先级
        'Sub': 2,  # 双目算术运算符，中等优先级
        'Mul': 3,  # 双目算术运算符，较高优先级
        'Div': 3,  # 双目算术运算符，较高优先级
        'Greater': 1,  # 比较运算符，较低优先级
        'Less': 1,  # 比较运算符，较低优先级
        'Ref': 5,  # 时间序列单目运算符，最高优先级
        'Mean': 4,  # 时间序列单目运算符，较高优先级
        'Sum': 4,  # 时间序列单目运算符，较高优先级
        'Std': 4,  # 时间序列单目运算符，较高优先级
        'Var': 4,  # 时间序列单目运算符，较高优先级
        'Max': 4,  # 时间序列单目运算符，较高优先级
        'Min': 4,  # 时间序列单目运算符，较高优先级
        'Med': 4,  # 时间序列单目运算符，较高优先级
        'Mad': 4,  # 时间序列单目运算符，较高优先级
        'Delta': 4,  # 时间序列单目运算符，较高优先级
        'WMA': 4,  # 时间序列单目运算符，较高优先级
        'EMA': 4,  # 时间序列单目运算符，较高优先级
        'Cov': 3,  # 时间序列双目运算符，较高优先级
        'Corr': 3  # 时间序列双目运算符，较高优先级
    }
    stack = []
    output = []

    # Split the expression into tokens using spaces around parentheses and commas.
    tokens = [t for t in expression.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').split()]

    for token in tokens:
        if token == '(':
            stack.append(token)
        elif token == ')':
            # Pop all elements until the matching '(' is found
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Pop the '('
            if stack and stack[-1] in precedence:
                # The token at the top of the stack is a function name
                output.append(stack.pop())
        elif token in precedence:
            while (stack and stack[-1] != '(' and
                   (stack[-1] in precedence and precedence[token] <= precedence[stack[-1]])):
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token)  # Directly append any other token which should be an operand.

    while stack:
        if stack[-1] == '(':
            raise ValueError("Mismatched parentheses")
        output.append(stack.pop())

    return [token for token in output if (token != ',' and token != 'Constant')]

def postfix_to_num_postfix(RPN_sequence):
    token2action_dict = {
    0: 'Abs',
    1: 'Log',
    2: 'Add',
    3: 'Sub',
    4: 'Mul',
    5: 'Div',
    6: 'Greater',
    7: 'Less',
    8: 'Ref',
    9: 'Mean',
    10: 'Sum',
    11: 'Std',
    12: 'Var',
    13: 'Max',
    14: 'Min',
    15: 'Med',
    16: 'Mad',
    17: 'Delta',
    18: 'WMA',
    19: 'EMA',
    20: 'Cov',
    21: 'Corr',
    22: '$open',
    23: '$close',
    24: '$high',
    25: '$low',
    26: '$volume',
    27: '$vwap',
    28: '10',
    29: '20',
    30: '30',
    31: '40',
    32: '50',
    33: '-30.0',
    34: '-10.0',
    35: '-5.0',
    36: '-2.0',
    37: '-1.0',
    38: '-0.5',
    39: '-0.01',
    40: '0.01',
    41: '0.5',
    42: '1.0',
    43: '2.0',
    44: '5.0',
    45: '10.0',
    46: '30.0',
    47: 'SEP'}

    num_sequence = []
    # 遍历输入列表
    for item in RPN_sequence:
        # 寻找匹配的键
        for key, values in token2action_dict.items():
            if isinstance(values, list):
                if item in values:
                    num_sequence.append(key)
                    break
            elif values == item:
                num_sequence.append(key)
                break
        else:
            # 如果找不到匹配项，则抛出异常或者处理这种情况（例如，可以打印警告信息）
            print(f"Warning: No match found for {item}")
    num_sequence.append(47)
    return num_sequence

def postfix_to_infix(expression_list):
    stack = []

    for token in expression_list:
        if token not in ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Abs', 'Log', 'Ref', 'Mean', 'Sum', 'Std', 'Var', 'Max', 'Min', 'Med', 'Mad', 'Delta', 'WMA', 'EMA', 'Cov', 'Corr']:
            # 如果是操作数，直接压入栈中
            stack.append(token)
        else:
            # 如果是操作符，根据操作符的类型处理操作数
            if token in ['Abs', 'Log', 'Ref', 'WMA', 'EMA']:
                # 一元操作符
                if token == 'Ref':
                    # Ref 需要两个参数
                    second_operand = stack.pop()
                    first_operand = stack.pop()
                    new_expr = f"Ref({first_operand}, {second_operand})"
                elif token in ['WMA', 'EMA']:
                    # WMA 和 EMA 需要两个参数
                    second_operand = stack.pop()
                    first_operand = stack.pop()
                    new_expr = f"{token}({first_operand}, {second_operand})"
                else:
                    # 其他一元操作符
                    operand = stack.pop()
                    new_expr = f"{token}({operand})"
                stack.append(new_expr)
            elif token in ['Mean', 'Sum', 'Std', 'Var', 'Max', 'Min', 'Med', 'Mad']:
                # 这些操作符需要两个参数
                second_operand = stack.pop()
                first_operand = stack.pop()
                new_expr = f"{token}({first_operand}, {second_operand})"
                stack.append(new_expr)
            elif token in ['Cov', 'Corr']:
                # 这些操作符需要三个参数
                third_operand = stack.pop()
                second_operand = stack.pop()
                first_operand = stack.pop()
                new_expr = f"{token}({first_operand}, {second_operand}, {third_operand})"
                stack.append(new_expr)
            elif token in ['Delta']:
                # Delta 需要两个参数
                second_operand = stack.pop()
                first_operand = stack.pop()
                new_expr = f"Delta({first_operand}, {second_operand})"
                stack.append(new_expr)
            else:
                # 二元操作符
                operand2 = stack.pop()
                operand1 = stack.pop()
                new_expr = f"{token}({operand1}, {operand2})"
                stack.append(new_expr)

    if len(stack) != 1:
        return None
    else:
        return stack[0].replace(' ', '')





def num_postfix_to_postfix(num_sequence):
    action2token_dict = {
    'Abs': 0,
    'Log': 1,
    'Add': 2,
    'Sub': 3,
    'Mul': 4,
    'Div': 5,
    'Greater': 6,
    'Less': 7,
    'Ref': 8,
    'Mean': 9,
    'Sum': 10,
    'Std': 11,
    'Var': 12,
    'Max': 13,
    'Min': 14,
    'Med': 15,
    'Mad': 16,
    'Delta': 17,
    'WMA': 18,
    'EMA': 19,
    'Cov': 20,
    'Corr': 21,
    '$open': 22,
    '$close': 23,
    '$high': 24,
    '$low': 25,
    '$volume': 26,
    '$vwap': 27,
    '10': 28,
    '20': 29,
    '30': 30,
    '40': 31,
    '50': 32,
    '-30.0': 33,
    '-10.0': 34,
    '-5.0': 35,
    '-2.0': 36,
    '-1.0': 37,
    '-0.5': 38,
    '-0.01': 39,
    '0.01': 40,
    '0.5': 41,
    '1.0': 42,
    '2.0': 43,
    '5.0': 44,
    '10.0': 45,
    '30.0': 46,
    'SEP': 47}

    # 创建逆向映射
    num2token_dict = {v: k for k, v in action2token_dict.items()}

    # 转换数字序列回字符序列
    char_sequence = []
    for num in num_sequence:
        if num in num2token_dict:
            token = num2token_dict[num]
            if token != 'SEP':
                char_sequence.append(token)
        else:
            # 如果找不到匹配项，则抛出异常或者处理这种情况（例如，可以打印警告信息）
            print(f"Warning: No match found for number {num}")

    return char_sequence

# 定义计算IC的函数
def calculate_ic(df1, df2):
    # 确保两个DataFrame在日期和代码上对齐
    merged_df = pd.merge(df1, df2, on=['date', 'code'], suffixes=('_policy', '_expert'))
    merged_df = merged_df.dropna(subset=['value_policy', 'value_expert'])
    merged_df = merged_df[np.isfinite(merged_df['value_policy']) & np.isfinite(merged_df['value_expert'])]

    ic, _ = pearsonr(merged_df['value_policy'], merged_df['value_expert'])
    return ic

if __name__ == '__main__':
    expression = "Mean(Mul(Constant(10.0),Div($close,$vwap)),20)"
    expression = "Delta(Std(Mul(Constant(5.0),$open),40),20)"
    expression = "Mul(Constant(-1.0), WMA(Div($close,$low), 20))"
    expression = "Div(Constant(1.0), Sub(WMA($volume, 20), Constant(10.0)))"
    expression = "Div(Greater(Greater(Constant(-10.0), Add(Add($high, Constant(-5.0)), Constant(2.0))), WMA($low, 50)), $low)"
    expression = "Log(Div(Sub(Add(Greater(Std($vwap, 30), $close), Constant(1.0)), Sub(Constant(1.0),Abs(EMA($low, 50)))), $close))"
    expression = "Std(EMA(Max($high, 10), 30), 40)"


    postfix_expression = infix_to_postfix(expression)
    postfix_num_expression = postfix_to_num_postfix(postfix_expression)
    infix_expression = postfix_to_infix(postfix_expression)
    print(infix_expression)
    print(postfix_expression)
    print(postfix_num_expression)
    postfix_expression = num_postfix_to_postfix(postfix_num_expression)
    print(postfix_expression)









