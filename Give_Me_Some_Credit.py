# Give_Me_Some_Credit.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os
import warnings

# -------------------------- 全局配置（彻底消除所有警告） --------------------------
warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated")
warnings.filterwarnings("ignore", message="Invalid property specified")
st.set_page_config(page_title="任务1 - Give Me Some Credit", page_icon="📈", layout="wide")
st.title("📈 任务 1: Give Me Some Credit 实验成果 ")
st.markdown("---")


# -------------------------- 数据加载 --------------------------
@st.cache_data(show_spinner="正在加载实验数据...")
def load_experiment_data():
    with open("./output_assets/model_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


try:
    res_data = load_experiment_data()
except FileNotFoundError:
    st.error("❌ 未找到实验数据文件！请先在终端运行 `python run_experiment.py` 生成真实数据。")
    st.stop()

# -------------------------- 第一部分：可视化成果展示 --------------------------
st.header("📊 第一部分：可视化成果展示")
tab1, tab2, tab3 = st.tabs(["1. 特征 IV 值分析", "2. 模型 ROC / PR 曲线对比", "3. 创新模型 KS 曲线"])

with tab1:
    st.subheader("💡 各特征信息值 (Information Value) 真实解算分布")
    iv_df = pd.DataFrame({
        "特征名称": res_data["iv_data"]["features"],
        "真实IV值": res_data["iv_data"]["iv_values"]
    }).sort_values(by="真实IV值", ascending=False)
    fig_iv = px.bar(iv_df, x="特征名称", y="真实IV值", text_auto='.3f',
                    title="现场公式计算得到的真实特征区分度排行 (IV)",
                    color="真实IV值", color_continuous_scale="Blues")
    st.plotly_chart(fig_iv, width='stretch', config={})

with tab2:
    st.subheader("🤝 5大模型五折交叉验证真实泛化性能对比 (ROC / PR)")
    c1, c2 = st.columns(2)
    fig_roc, fig_pr = go.Figure(), go.Figure()

    # ✅ 修复：opacity移到Scatter顶层，line只保留width
    line_config = dict(width=2)
    for model_name, curves in res_data["curves"].items():
        fig_roc.add_trace(
            go.Scatter(x=curves["fpr"], y=curves["tpr"],
                       name=f"{model_name} (AUC={curves['auc_value']:.4f})",
                       mode='lines', line=line_config, opacity=0.8))
        fig_pr.add_trace(go.Scatter(x=curves["recall"], y=curves["precision"],
                                    name=f"{model_name}", mode='lines',
                                    line=line_config, opacity=0.8))

    fig_roc.update_layout(title="ROC 曲线 (越靠近左上方越优秀)",
                          xaxis_title="假正率 (FPR)", yaxis_title="真正率 (TPR)")
    fig_pr.update_layout(title="PR 曲线",
                         xaxis_title="召回率 (Recall)", yaxis_title="精准率 (Precision)")

    with c1: st.plotly_chart(fig_roc, width='stretch', config={})
    with c2: st.plotly_chart(fig_pr, width='stretch', config={})

with tab3:
    st.subheader("🔍 最佳创新模型现场真实 KS 曲线")
    # ✅ 修复：和run_experiment.py完全一致的模型名称
    innovate_model = "Focal-Loss权重优调 LightGBM (创新改进)"

    if innovate_model in res_data["curves"]:
        target_curve = res_data["curves"][innovate_model]
        x_axis = list(range(len(target_curve["fpr"])))
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=x_axis, y=target_curve["tpr"],
                                    name="正样本累积比例 (TPR)", mode='lines',
                                    line=dict(color='green', width=2)))
        fig_ks.add_trace(go.Scatter(x=x_axis, y=target_curve["fpr"],
                                    name="负样本累积比例 (FPR)", mode='lines',
                                    line=dict(color='red', width=2)))
        max_idx = target_curve["max_ks_idx"]
        fig_ks.add_shape(type="line", x0=max_idx, y0=target_curve["fpr"][max_idx],
                         x1=max_idx, y1=target_curve["tpr"][max_idx],
                         line=dict(color="blue", width=3, dash="dot"))
        fig_ks.update_layout(title=f"{innovate_model} 的真实 KS 曲线 (Max KS = {target_curve['ks_value']:.4f})",
                             xaxis_title="样本百分位划分", yaxis_title="累积占比")
        st.plotly_chart(fig_ks, width='stretch', config={})
    else:
        st.error(f"❌ 模型 {innovate_model} 不存在，请检查run_experiment.py中的模型名称")

# --- 第二部分：5大算法对比表格 ---
st.markdown("---")
st.header("🤖 第二部分：算法四维性能对比表格")
rows = []
for model_name, curves in res_data["curves"].items():
    rows.append([
        model_name,
        f"{curves['auc_value']:.4f}",
        f"{curves['ks_value']:.4f}",
        f"{curves['train_time']} 秒",
        "🔥 核心创新" if "创新改进" in model_name else "否"
    ])
df_compare = pd.DataFrame(rows, columns=["算法模型体系", "五折验证 AUC", "风控区分度 (KS值)", "五折真实运行耗时",
                                         "是否为创新模型"])
st.dataframe(df_compare, width='stretch')

# --- 第三部分：在线模拟风控预测沙盒 ---
st.markdown("---")
st.header("🔮 第三部分：高分加分项 —— 客户违约风险实时在线评估沙盒")
st.markdown(
    "👉 **业务场景验证**：在下方手动调整新申请授信客户的财务与信用历史指标，一键触发后台保存的创新改进模型，实时输出该客户的风险概率。")

c1, col2, col3 = st.columns(3)
with c1:
    revol = st.slider("可用授信额度利用率 [RevolvingUtilizationOfUnsecuredLines]", 0.0, 2.0, 0.3, 0.05)
    age = st.number_input("申请人年龄 [age]", min_value=18, max_value=100, value=45)
    past_due_30 = st.slider("历史逾期 30-59 天次数 [NumberOfTime30-59DaysPastDueNotWorse]", 0, 10, 0)
with col2:
    debt_ratio = st.number_input("负债率 [DebtRatio]", min_value=0.0, value=0.35)
    monthly_income = st.number_input("月收入总额 ($) [MonthlyIncome]", min_value=0, value=6500)
    past_due_60 = st.slider("历史逾期 60-89 天次数 [NumberOfTime60-89DaysPastDueNotWorse]", 0, 10, 0)
with col3:
    open_loans = st.number_input("未结清贷款及信用卡笔数 [NumberOfOpenCreditLinesAndLoans]", min_value=0, value=8)
    real_estate = st.number_input("不动产贷款或房产笔数 [NumberRealEstateLoansOrLines]", min_value=0, value=1)
    past_due_90 = st.slider("历史逾期 90 天及以上严重次数 [NumberOfTimes90DaysLate]", 0, 15, 0)
    dependents = st.number_input("家属抚养人数 [NumberOfDependents]", min_value=0, value=0)

if st.button("🎯 立即调阅离线固化模型生成授信风险报告", width='stretch'):
    model_pkl_path = "./output_assets/best_task1_model.pkl"
    if os.path.exists(model_pkl_path):
        with open(model_pkl_path, 'rb') as f:
            clf = pickle.load(f)
        task1_features = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ]
        input_row = pd.DataFrame(
            [[revol, age, past_due_30, debt_ratio, monthly_income, open_loans, past_due_90, real_estate, past_due_60,
              dependents]],
            columns=task1_features
        )
        prob = clf.predict_proba(input_row)[0, 1]
        st.markdown("---")
        st.subheader("📝 现场实时评级报告结论：")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(label="该客户模型预测违约概率 (PD)", value=f"{prob * 100:.2f}%")
        with cc2:
            if prob < 0.06:
                st.success("✅ 评级：A级 (系统评估极安全) —— 建议予以批量放款！")
            elif prob < 0.25:
                st.warning("⚠️ 评级：C级 (中度潜在风险) —— 建议要求人工辅助流水证明。")
            else:
                st.error("❌ 评级：E级 (触发风控拦截红线) —— 严重逾期倾向，系统直接秒拒！")
    else:
        st.error("未找到保存的模型文件，请重新运行后台 `run_experiment.py` 脚本。")