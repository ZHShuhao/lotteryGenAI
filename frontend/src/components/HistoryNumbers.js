// import React, { useState, useEffect } from "react";
// import axios from "axios";
// import "./HistoryNumbers.css";
//
// const HistoryNumbers = () => {
//   const [data, setData] = useState([]);
//   const [currentPage, setCurrentPage] = useState(1);
//   const [loading, setLoading] = useState(true);
//   const [lotteryType, setLotteryType] = useState("MegaMillion"); // 默认显示 Mega Million 数据
//
//   const itemsPerPage = 20;
//   const maxPageButtons = 10;
//
//   // 动态 API URL，根据 lotteryType 更新
//   const apiUrl = `http://127.0.0.1:5000/api/history-numbers/${lotteryType}`;
//
//   useEffect(() => {
//     setLoading(true); // 切换数据时显示加载动画
//     axios
//       .get(apiUrl)
//       .then((response) => {
//         setData(response.data);
//         setLoading(false);
//       })
//       .catch((error) => {
//         console.error("Error fetching data:", error);
//         setLoading(false);
//       });
//   }, [apiUrl]);
//
//   const indexOfLastItem = currentPage * itemsPerPage;
//   const indexOfFirstItem = indexOfLastItem - itemsPerPage;
//   const currentData = data.slice(indexOfFirstItem, indexOfLastItem);
//
//   const totalPages = Math.ceil(data.length / itemsPerPage);
//
//   const startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
//   const endPage = Math.min(totalPages, startPage + maxPageButtons - 1);
//
//   const pageNumbers = Array.from(
//     { length: endPage - startPage + 1 },
//     (_, index) => startPage + index
//   );
//
//   const handlePageChange = (pageNumber) => {
//     if (pageNumber >= 1 && pageNumber <= totalPages) {
//       setCurrentPage(pageNumber);
//     }
//   };
//
//   const handleInputPageChange = (e) => {
//     const page = parseInt(e.target.value, 10);
//     if (page >= 1 && page <= totalPages) {
//       setCurrentPage(page);
//     }
//   };
//
//   const handleLotteryTypeChange = (type) => {
//     setLotteryType(type);
//     setCurrentPage(1); // 切换数据时回到第一页
//   };
//
//   if (loading) {
//     return <div className="loading">Loading...</div>;
//   }
//
//   return (
//     <div className="history-numbers">
//       <h1>{lotteryType === "MegaMillion" ? "Mega Million Win History" : "Power Ball Win History"}</h1>
//       <div className="button-group">
//         <button
//           className={`lottery-button ${lotteryType === "MegaMillion" ? "active" : ""}`}
//           onClick={() => handleLotteryTypeChange("MegaMillion")}
//         >
//           Mega Ball
//         </button>
//         <button
//           className={`lottery-button ${lotteryType === "PowerBall" ? "active" : ""}`}
//           onClick={() => handleLotteryTypeChange("PowerBall")}
//         >
//           Power Ball
//         </button>
//       </div>
//       <div className="history-table-wrapper">
//         <table className="history-table">
//           <thead>
//             <tr>
//               <th>Date</th>
//               <th>Number1</th>
//               <th>Number2</th>
//               <th>Number3</th>
//               <th>Number4</th>
//               <th>Number5</th>
//               <th>{lotteryType === "MegaMillion" ? "Mega Ball" : "Power Ball"}</th>
//               <th>Megaplier</th>
//               <th>Jackpot</th>
//             </tr>
//           </thead>
//           <tbody>
//             {currentData.map((draw, index) => (
//               <tr key={index}>
//                 <td>{draw.DrawingDate}</td>
//                 <td>{draw.Number1}</td>
//                 <td>{draw.Number2}</td>
//                 <td>{draw.Number3}</td>
//                 <td>{draw.Number4}</td>
//                 <td>{draw.Number5}</td>
//                 <td>{draw.MegaBall || draw.PowerBall}</td>
//                 <td>{draw.Megaplier}</td>
//                 <td>{draw.JackPot}</td>
//               </tr>
//             ))}
//           </tbody>
//         </table>
//       </div>
//       <div className="pagination">
//         <button
//           className="page-button"
//           onClick={() => handlePageChange(currentPage - 1)}
//           disabled={currentPage === 1}
//         >
//           Previous
//         </button>
//         {pageNumbers.map((number) => (
//           <button
//             key={number}
//             className={`page-button ${currentPage === number ? "active" : ""}`}
//             onClick={() => handlePageChange(number)}
//           >
//             {number}
//           </button>
//         ))}
//         <button
//           className="page-button"
//           onClick={() => handlePageChange(currentPage + 1)}
//           disabled={currentPage === totalPages}
//         >
//           Next
//         </button>
//         <div className="jump-to-page">
//           <span>Jump to page: </span>
//           <input
//             type="number"
//             min="1"
//             max={totalPages}
//             onChange={handleInputPageChange}
//             placeholder="Enter page"
//           />
//         </div>
//       </div>
//     </div>
//   );
// };
//
// export default HistoryNumbers;


//动态显示mega 和 power 2种格式

// import React, { useState, useEffect } from "react";
// import axios from "axios";
// import "./HistoryNumbers.css";
//
// const HistoryNumbers = () => {
//   const [data, setData] = useState([]);
//   const [currentPage, setCurrentPage] = useState(1);
//   const [loading, setLoading] = useState(true);
//   const [lotteryType, setLotteryType] = useState("MegaMillion"); // 默认显示 Mega Million 数据
//
//   const itemsPerPage = 20;
//   const maxPageButtons = 10;
//
//   // 动态 API URL，根据 lotteryType 更新
//   const apiUrl = `http://127.0.0.1:5000/api/history-numbers/${lotteryType}`;
//
//   // 动态表头和字段
//   const tableConfig = {
//     MegaMillion: {
//       headers: ["Date", "Number1", "Number2", "Number3", "Number4", "Number5", "Mega Ball", "Megaplier", "Jackpot"],
//       fields: ["DrawingDate", "Number1", "Number2", "Number3", "Number4", "Number5", "MegaBall", "Megaplier", "JackPot"],
//     },
//     PowerBall: {
//       headers: ["Date", "Number1", "Number2", "Number3", "Number4", "Number5", "Power Ball", "Jackpot", "Cash Value"],
//       fields: ["DrawingDate", "Number1", "Number2", "Number3", "Number4", "Number5", "PowerBall", "JackPot", "EstimatedCashValue"],
//     },
//   };
//
//   useEffect(() => {
//     setLoading(true); // 切换数据时显示加载动画
//     axios
//       .get(apiUrl)
//       .then((response) => {
//         setData(response.data);
//         setLoading(false);
//       })
//       .catch((error) => {
//         console.error("Error fetching data:", error);
//         setLoading(false);
//       });
//   }, [apiUrl]);
//
//   const indexOfLastItem = currentPage * itemsPerPage;
//   const indexOfFirstItem = indexOfLastItem - itemsPerPage;
//   const currentData = data.slice(indexOfFirstItem, indexOfLastItem);
//
//   const totalPages = Math.ceil(data.length / itemsPerPage);
//
//   const startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
//   const endPage = Math.min(totalPages, startPage + maxPageButtons - 1);
//
//   const pageNumbers = Array.from(
//     { length: endPage - startPage + 1 },
//     (_, index) => startPage + index
//   );
//
//   const handlePageChange = (pageNumber) => {
//     if (pageNumber >= 1 && pageNumber <= totalPages) {
//       setCurrentPage(pageNumber);
//     }
//   };
//
//   const handleInputPageChange = (e) => {
//     const page = parseInt(e.target.value, 10);
//     if (page >= 1 && page <= totalPages) {
//       setCurrentPage(page);
//     }
//   };
//
//   const handleLotteryTypeChange = (type) => {
//     setLotteryType(type);
//     setCurrentPage(1); // 切换数据时回到第一页
//   };
//
//   if (loading) {
//     return <div className="loading">Loading...</div>;
//   }
//
//   const { headers, fields } = tableConfig[lotteryType];
//
//   return (
//     <div className="history-numbers">
//       <h1>{lotteryType === "MegaMillion" ? "Mega Million Win History" : "Power Ball Win History"}</h1>
//       <div className="button-group">
//         <button
//           className={`lottery-button ${lotteryType === "MegaMillion" ? "active" : ""}`}
//           onClick={() => handleLotteryTypeChange("MegaMillion")}
//         >
//           Mega Ball
//         </button>
//         <button
//           className={`lottery-button ${lotteryType === "PowerBall" ? "active" : ""}`}
//           onClick={() => handleLotteryTypeChange("PowerBall")}
//         >
//           Power Ball
//         </button>
//       </div>
//       <div className="history-table-wrapper">
//         <table className="history-table">
//           <thead>
//             <tr>
//               {headers.map((header, index) => (
//                 <th key={index}>{header}</th>
//               ))}
//             </tr>
//           </thead>
//           <tbody>
//             {currentData.map((draw, index) => (
//               <tr key={index}>
//                 {fields.map((field, fieldIndex) => (
//                   <td key={fieldIndex}>{draw[field] || "N/A"}</td>
//                 ))}
//               </tr>
//             ))}
//           </tbody>
//         </table>
//       </div>
//       <div className="pagination">
//         <button
//           className="page-button"
//           onClick={() => handlePageChange(currentPage - 1)}
//           disabled={currentPage === 1}
//         >
//           Previous
//         </button>
//         {pageNumbers.map((number) => (
//           <button
//             key={number}
//             className={`page-button ${currentPage === number ? "active" : ""}`}
//             onClick={() => handlePageChange(number)}
//           >
//             {number}
//           </button>
//         ))}
//         <button
//           className="page-button"
//           onClick={() => handlePageChange(currentPage + 1)}
//           disabled={currentPage === totalPages}
//         >
//           Next
//         </button>
//         <div className="jump-to-page">
//           <span>Jump to page: </span>
//           <input
//             type="number"
//             min="1"
//             max={totalPages}
//             onChange={handleInputPageChange}
//             placeholder="Enter page"
//           />
//         </div>
//       </div>
//     </div>
//   );
// };
//
// export default HistoryNumbers;

import React, { useState, useEffect } from "react";
import axios from "axios";
import "./HistoryNumbers.css";
import API from "../api"; // 路径根据实际结构来改

const HistoryNumbers = () => {
  const [data, setData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [lotteryType, setLotteryType] = useState("MegaMillion"); // 默认 Mega Million

  const itemsPerPage = 20;
  const maxPageButtons = 10;

  // // 动态 API URL
  // const apiUrl = `http://127.0.0.1:5000/api/history-numbers/${lotteryType}`;

   // 异步加载数据
   useEffect(() => {
    const fetchHistoryNumbers = async () => {
      try {
        setLoading(true);
        const response = await API.get(`/api/history-numbers/${lotteryType}`);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching history numbers:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchHistoryNumbers();
  }, [lotteryType]);

  // 动态表头和字段配置
  const tableConfig = {
    MegaMillion: {
      headers: ["Date", "Number1", "Number2", "Number3", "Number4", "Number5", "Mega Ball", "Megaplier", "Jackpot"],
      fields: ["DrawingDate", "Number1", "Number2", "Number3", "Number4", "Number5", "MegaBall", "Megaplier", "JackPot"],
    },
    PowerBall: {
      headers: ["Date", "Number1", "Number2", "Number3", "Number4", "Number5", "Power Ball", "Jackpot", "Estimated Cash Value"],
      fields: ["DrawingDate", "Number1", "Number2", "Number3", "Number4", "Number5", "PowerBall", "Jackpot", "EstimatedCashValue"],
    },
  };

  // useEffect(() => {
  //   setLoading(true);
  //   axios
  //     .get(apiUrl)
  //     .then((response) => {
  //       setData(response.data);
  //       setLoading(false);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching data:", error);
  //       setLoading(false);
  //     });
  // }, [apiUrl]);

  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentData = data.slice(indexOfFirstItem, indexOfLastItem);

  const totalPages = Math.ceil(data.length / itemsPerPage);

  const startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
  const endPage = Math.min(totalPages, startPage + maxPageButtons - 1);

  const pageNumbers = Array.from(
    { length: endPage - startPage + 1 },
    (_, index) => startPage + index
  );

  const handlePageChange = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setCurrentPage(pageNumber);
    }
  };

  const handleInputPageChange = (e) => {
    const page = parseInt(e.target.value, 10);
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  const handleLotteryTypeChange = (type) => {
    setLotteryType(type);
    setCurrentPage(1); // 切换数据时回到第一页
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  const { headers, fields } = tableConfig[lotteryType];

  return (
    <div className="history-numbers">
      <h1>{lotteryType === "MegaMillion" ? "Mega Million Win History" : "Power Ball Win History"}</h1>
      <div className="button-group">
        <button
          className={`lottery-button ${lotteryType === "MegaMillion" ? "active" : ""}`}
          onClick={() => handleLotteryTypeChange("MegaMillion")}
        >
          Mega Ball
        </button>
        <button
          className={`lottery-button ${lotteryType === "PowerBall" ? "active" : ""}`}
          onClick={() => handleLotteryTypeChange("PowerBall")}
        >
          Power Ball
        </button>
      </div>
      <div className="history-table-wrapper">
        <table className="history-table">
          <thead>
            <tr>
              {headers.map((header, index) => (
                <th key={index}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {currentData.map((draw, index) => (
              <tr key={index}>
                {fields.map((field, fieldIndex) => (
                  <td key={fieldIndex}>{draw[field] || "N/A"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button
          className="page-button"
          onClick={() => handlePageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        {pageNumbers.map((number) => (
          <button
            key={number}
            className={`page-button ${currentPage === number ? "active" : ""}`}
            onClick={() => handlePageChange(number)}
          >
            {number}
          </button>
        ))}
        <button
          className="page-button"
          onClick={() => handlePageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
        <div className="jump-to-page">
          <span>Jump to page: </span>
          <input
            type="number"
            min="1"
            max={totalPages}
            onChange={handleInputPageChange}
            placeholder="Enter page"
          />
        </div>
      </div>
    </div>
  );
};

export default HistoryNumbers;

