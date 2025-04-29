import React, { useEffect, useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from "chart.js";
import "./HistoryStatistic.css";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

const HistoryStatistic = () => {
  const [statisticsType, setStatisticsType] = useState("MegaMillion"); // 当前统计类型
  const [whiteBallOccurrences, setWhiteBallOccurrences] = useState({}); // 白球数据
  const [specialBallOccurrences, setSpecialBallOccurrences] = useState({}); // 特殊球（Mega或Power）数据
  const [loading, setLoading] = useState(true); // 加载状态
  const [error, setError] = useState(null); // 错误状态

  // 数据加载函数
  const fetchStatisticsData = async (type) => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(
        `http://127.0.0.1:5000/api/history-statistic/${type}` // 动态获取API
      );
      const data = response.data;

       // 绑定白球数据
    setWhiteBallOccurrences(data.whiteballoccurrences || {});

      // 根据统计类型动态绑定特殊球数据
      if (type === "MegaMillion") {
        setSpecialBallOccurrences(data.megaBalloccurrences || {});
      } else if (type === "PowerBall") {
        setSpecialBallOccurrences(data.powerballoccurrences || {}); // 修正为 powerballoccurrences
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      setError("Failed to fetch data. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  // 当统计类型变化时重新加载数据
  useEffect(() => {
    fetchStatisticsData(statisticsType);
  }, [statisticsType]);

  // 切换统计类型按钮
  const handleButtonClick = (type) => {
    setStatisticsType(type);
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  // 图表选项
  const chartOptions = (xTitle, yTitle) => ({
    responsive: true,
    maintainAspectRatio: false,
    scales: {
       x: {
      title: {
        display: true, // 显示 X 轴标题
        text: xTitle,
        color: "white", // 标题颜色
        font: {
          size: 14, // 字体大小
        },
        padding: {
          top: 3, // 标题与轴之间的间距
        },
      },
      ticks: {
        color: "white", // X 轴刻度标签颜色
        font: {
          size: 12, // 字体大小
        },
        padding: 3, // 标签与轴线之间的间距
      },
      grid: {
        color: "rgba(255, 255, 255, 0.2)", // 网格线颜色
      },
      position: "bottom", // 确保 X 轴在容器底部
    },
      y: {
        title: {
          display: true,
          text: yTitle,
          color: "white",
        },
        ticks: {
          color: "white",
        },
      },
    },
    plugins: {
      legend: {
        labels: {
          color: "white",
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => `Frequency: ${context.raw}`,
        },
      },
    },
  });

  // 白球图表数据
  const whiteBallData = {
    labels: Object.keys(whiteBallOccurrences),
    datasets: [
      {
        label: `${statisticsType === "MegaMillion" ? "White Ball Frequency" : "White Ball Frequency"}`,
        data: Object.values(whiteBallOccurrences),
        backgroundColor: statisticsType === "MegaMillion" ? "rgba(255, 204, 0, 0.8)" : "rgba(255, 0, 0, 0.8)", // 动态颜色
        hoverBackgroundColor: statisticsType === "MegaMillion" ? "rgba(255, 204, 0, 1)" : "rgba(255, 0, 0, 1)",
        borderColor: statisticsType === "MegaMillion" ? "rgba(255, 204, 0, 1)" : "rgba(255, 0, 0, 1)",
        borderWidth: 1,
      },
    ],
  };

  // 特殊球图表数据
  const specialBallData = {
    labels: Object.keys(specialBallOccurrences),
    datasets: [
      {
        label: `${statisticsType === "MegaMillion" ? "Mega Ball Frequency" : "Power Ball Frequency"}`,
        data: Object.values(specialBallOccurrences),
        backgroundColor: statisticsType === "MegaMillion" ? "rgba(75, 192, 192, 0.8)" : "rgba(255, 255, 255, 0.8)", // MegaMillion 使用原颜色，PowerBall 使用白色
        hoverBackgroundColor: statisticsType === "MegaMillion" ? "rgba(75, 192, 192, 1)" : "rgba(255, 255, 255, 1)",
        borderColor: statisticsType === "MegaMillion" ? "rgba(75, 192, 192, 1)" : "rgba(255, 255, 255, 1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="history-statistic">
      <h1>Lottery History Frequency Statistics</h1>
      <div className="history-button-container">
        <button
          className={`toggle-button ${statisticsType === "MegaMillion" ? "active" : ""}`}
          onClick={() => handleButtonClick("MegaMillion")}
        >
          Mega Million
        </button>
        <button
          className={`toggle-button ${statisticsType === "PowerBall" ? "active" : ""}`}
          onClick={() => handleButtonClick("PowerBall")}
        >
          Power Ball
        </button>
      </div>
      <div className="chart-container">
        <h2>White Ball Frequencies</h2>
        <Bar data={whiteBallData} options={chartOptions("Ball Number", "Frequency")} />
      </div>
      <div className="chart-container">
        <h2>{statisticsType === "MegaMillion" ? "Mega Ball Frequencies" : "Power Ball Frequencies"}</h2>
        <Bar data={specialBallData} options={chartOptions("Ball Number", "Frequency")} />
      </div>
    </div>
  );
};

export default HistoryStatistic;



