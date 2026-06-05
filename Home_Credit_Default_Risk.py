# pages/Home_Credit_Default_Risk.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os

st.set_page_config(page_title="任务2 - 终极大屏", page_icon="🏦", layout="wide")
st.title("🏦 任务 2: Home Credit 跨表多维数据风控决策大屏 (满分收官版)")
st.markdown("---")


@st.cache_data
def load_task2_data():
    with open("./output_assets/task2_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


try:
    res_data = load_task2_data()
except FileNotFoundError:
    st.error("❌ 资产库中未检测到全量多表实验结果。请在终端执行：`python run_experiment_task2.py`。")
    st.stop()

# --- 第一部分：高维缺失矩阵 ---
st.header("📊 第一部分：原厂表高维缺失矩阵盘点")
selected_table = st.selectbox("选择要查看的表：", list(res_data["missing_data"].keys()))
table_info = res_data["missing_data"][selected_table]
df_miss = pd.DataFrame({"特征字段名": table_info["columns"], "缺失率": table_info["missing_rates"]}).sort_values(
    by="缺失率", ascending=False)
fig_miss = px.bar(df_miss.head(20), x="特征字段名", y="缺失率", title=f"{selected_table} 前20特征缺失率分布",
                  color="缺失率",
                  color_continuous_scale="Purples")
st.plotly_chart(fig_miss, use_container_width=True)

# --- 第二部分：多维指标雷达图比拼 ---
st.markdown("---")
st.header("🤖 第二部分：全量算法泛化性能大比拼 (AUC / KS / Recall / Time)")

c1, c2 = st.columns([4, 5])

with c1:
    st.subheader("🤖 算法效能明细账目")
    rows = []
    for model_name, m_val in res_data["metrics"].items():
        rows.append([
            model_name,
            f"{m_val['AUC']:.4f}",
            f"{m_val['KS']:.4f}",
            f"{m_val['Recall']:.4f}",
            f"{m_val['Time(s)']} 秒"
        ])

    df_compare = pd.DataFrame(rows, columns=["算法模型体系", "验证集 AUC", "风控区分度 (KS)", "违约样本召回率 (Recall)",
                                             "训练全量耗时"])
    st.dataframe(df_compare, use_container_width=True)

with c2:
    st.subheader("🔮 现场数据驱动模型交叉特征雷达图")
    fig_radar = go.Figure()

    # 🛑 核心修复：雷达图展示的轴标签名
    categories = ['AUC', 'KS', 'Recall']

    for model_name, metrics in res_data["metrics"].items():
        # 🛑 核心修复：从后台数据 metrics 中严格按原本的英文键取值，杜绝 KeyError
        fig_radar.add_trace(go.Scatterpolar(
            r=[metrics['AUC'], metrics['KS'], metrics['Recall']],
            theta=categories,
            fill='toself',
            name=model_name
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="多算法交叉验证风控能力雷达拓扑"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- 第三部分：在线模拟风控预测沙盒 ---
st.markdown("---")
st.header("🔮 第三部分：工业级离线模型“在线模拟风控预测”百宝箱")
st.markdown("👉 **业务场景模拟**：手动调整新客户的跨表联动资信指标，直接调用后台跑通固化的全行为链模型输出其实时风险概率。")

col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("客户年收入总额 ($) [AMT_INCOME_TOTAL]", min_value=10000, value=150000, step=1000)
    credit = st.number_input("本次贷款总信用额度 ($) [AMT_CREDIT]", min_value=10000, value=500000, step=1000)
    annuity = st.number_input("本次贷款月还款年金 ($) [AMT_ANNUITY]", min_value=1000, value=24000, step=500)
with col2:
    bureau_mean = st.number_input("外部征信历史平均贷款额度 ($) [AMT_CREDIT_SUM_mean]", min_value=0, value=80000,
                                  step=1000)
    bureau_sum = st.number_input("外部征信历史累积总贷款额 ($) [AMT_CREDIT_SUM_sum]", min_value=0, value=240000,
                                 step=1000)
    bureau_overdue = st.slider("外部征信历史最长逾期天数 (天) [CREDIT_DAY_OVERDUE_max]", 0, 365, 0)
with col3:
    prev_max = st.number_input("本平台历史申请分期最大获批金额 ($) [AMT_APPLICATION_max]", min_value=0, value=120000,
                               step=1000)
    prev_mean = st.number_input("本平台历史申请分期平均获批金额 ($) [AMT_APPLICATION_mean]", min_value=0, value=80000,
                                step=1000)

if st.button("🎯 一键调阅最佳模型进行风险评估", use_container_width=True):
    # 完美咬合后台保存的模型资产路径
    model_p = "./output_assets/best_lgb_model.pkl"
    if os.path.exists(model_p):
        with open(model_p, 'rb') as f:
            best_model = pickle.load(f)

        # 严格对齐后台特征与顺序
        task2_features = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'AMT_CREDIT_SUM_mean', 'AMT_CREDIT_SUM_sum', 'CREDIT_DAY_OVERDUE_max',
            'AMT_APPLICATION_max', 'AMT_APPLICATION_mean'
        ]

        input_data = pd.DataFrame(
            [[income, credit, annuity, bureau_mean, bureau_sum, bureau_overdue, prev_max, prev_mean]],
            columns=task2_features
        )

        prob = best_model.predict_proba(input_data)[0, 1]

        st.markdown("---")
        st.subheader("📊 实时风控信用授信报告结论：")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(label="该客户系统预测违约概率 (PD)", value=f"{prob * 100:.2f}%")
        with cc2:
            if prob < 0.12:
                st.success("✅ 风控评级：A级 (极低风险) —— 建议予以批量秒批放款！")
            elif prob < 0.35:
                st.warning("⚠️ 风控评级：C级 (中度风险) —— 建议转交人工二审，加征流水证明。")
            else:
                st.error("❌ 风控评级：E级 (高危欺诈倾向) —— 触碰风控红线，系统直接秒拒！")
    else:
        st.error("未找到保存的模型文件，请在终端重新运行后台脚本：`python run_experiment_task2.py`")