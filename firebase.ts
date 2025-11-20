// src/firebase.ts
import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDI3Y3LjP8QWJZ0JmIVqj7tkhw9bO6P9FM",
  authDomain: "smartseat-4b514.firebaseapp.com",
  projectId: "smartseat-4b514",
  storageBucket: "smartseat-4b514.appspot.com",
  messagingSenderId: "1075178711129",
  appId: "1:1075178711129:web:d2cca23dd8ebb060dca948",
  measurementId: "G-RGV89NZE7C",
};

// ✅ 防止 app 重复初始化
const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);

// ✅ 导出 services
export const auth = getAuth(app);
export const db = getFirestore(app);
