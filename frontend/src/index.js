// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';
// import App from './App';
// import reportWebVitals from './reportWebVitals';
//
// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );
//
// // If you want to start measuring performance in your app, pass a function
// // to log results (for example: reportWebVitals(console.log))
// // or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();

//粒子特效

// import React from "react";
// import ReactDOM from "react-dom/client";
// import "./index.css";
// import App from "./App";
// import { tsParticles } from "https://cdn.jsdelivr.net/npm/@tsparticles/engine@3.1.0/+esm";
// import { loadAll } from "https://cdn.jsdelivr.net/npm/@tsparticles/all@3.1.0/+esm";
//
// const root = ReactDOM.createRoot(document.getElementById("root"));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );
//
// // Load particle effects
// const configs = {
//   particles: {
//     number: {
//       value: 100,
//     },
//     color: {
//       value: "#2f0445",
//     },
//     links: {
//       enable: true,
//       distance: 300,
//     },
//     shape: {
//       type: "circle",
//     },
//     opacity: {
//       value: 1,
//     },
//     size: {
//       value: {
//         min: 4,
//         max: 6,
//       },
//     },
//     move: {
//       enable: true,
//       speed: 2,
//     },
//   },
//   background: {
//     color: "#9531e7",
//   },
// };
//
// async function loadParticles() {
//   await loadAll(tsParticles);
//   await tsParticles.load({ id: "tsparticles", options: configs });
// }
//
// loadParticles();



//彩色球
import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import { tsParticles } from "@tsparticles/engine";
import { loadAll } from "@tsparticles/all";

// import { tsParticles } from "https://cdn.jsdelivr.net/npm/@tsparticles/engine@3.1.0/+esm";
// import { loadAll } from "https://cdn.jsdelivr.net/npm/@tsparticles/all@3.1.0/+esm";


const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

const configs = {
  particles: {
    destroy: {
      mode: "split",
      split: {
        count: 1,
        factor: {
          value: {
            min: 2,
            max: 4
          }
        },
        rate: {
          value: 100
        },
        particles: {
          life: {
            count: 1,
            duration: {
              value: {
                min: 2,
                max: 3
              }
            }
          },
          move: {
            speed: {
              min: 10,
              max: 15
            }
          }
        }
      }
    },
    number: {
      value: 80
    },
    color: {
      value: [
        "#3998D0",
        "#2EB6AF",
        "#A9BD33",
        "#FEC73B",
        "#F89930",
        "#F45623",
        "#D62E32",
        "#EB586E",
        "#9952CF"
      ]
    },
    shape: {
      type: "circle"
    },
    opacity: {
      value: 1
    },
    size: {
      value: {
        min: 10,
        max: 15
      }
    },
    collisions: {
      enable: true,
      mode: "bounce"
    },
    move: {
      enable: true,
      speed: 3,
      outModes: "bounce"
    }
  },
  interactivity: {
    events: {
      onClick: {
        enable: true,
        mode: "pop"
      }
    }
  },
  background: {
    color: "#000000"
  }
};

// 确保页面缩放为 80%
window.onload = () => {
  document.body.style.zoom = "80%";
};

async function loadParticles(options) {
  await loadAll(tsParticles);

  await tsParticles.load({ id: "tsparticles", options });
}


loadParticles(configs);