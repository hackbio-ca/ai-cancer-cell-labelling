import Image from "next/image";

export default function Home() {
  return (
    <div className="flex justify-center items-center w-screen h-screen">
      <div className="w-80 h-60 bg-gray-100 flex justify-center items-center rounded-xl">
        <h1 className="font-bold">AI Cancer Image Detector</h1>
      </div>
    </div>
  );
}
