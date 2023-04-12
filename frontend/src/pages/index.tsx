/* eslint-disable @next/next/no-img-element */
import * as tf from "@tensorflow/tfjs";
import { Inter } from "next/font/google";
import { useRef, useState } from "react";
import { CanvasRef, Canvas } from "@/components/Canvas";
import { useFabricJs } from "@/hooks/useFabricJs";
import { useTensorflowModel } from "@/hooks/useTensorflowModel";
import { Spin } from "@/components/Spin";

const inter = Inter({ subsets: ["latin"] });

const maxIdx = (arr: number[]) => {
  return arr.indexOf(Math.max(...arr));
};

export default function Home() {
  const canvasRef = useRef<CanvasRef | null>(null);
  const { model, isLoading: isLoadingModel } =
    useTensorflowModel("/model.json");
  const [prediction, setPrediction] = useState<number>();
  useFabricJs(); // prefetch fabric-js

  const handleClick = async () => {
    const canvas = canvasRef?.current?.getCanvas();

    if (!canvas || !model) return;

    let image = tf.browser
      .fromPixels(
        canvas.getContext().getImageData(0, 0, canvas.width!, canvas.height!),
        1
      )
      .resizeBilinear([28, 28])
      .div(tf.scalar(255));
    const predictions = model.predict(image.reshape([1, 28, 28, 1]));
    if (
      "dataSync" in predictions &&
      typeof predictions.dataSync === "function"
    ) {
      setPrediction(maxIdx(Array.from(predictions.dataSync())));
    }
  };

  const clear = () => {
    canvasRef?.current?.getCanvas()?.clear?.();
    setPrediction(undefined);
  };

  if (isLoadingModel) {
    return (
      <div className="w-screen h-screen flex justify-center items-center">
        <Spin />
      </div>
    );
  }

  return (
    <main
      style={inter.style}
      className="w-screen h-screen flex justify-center items-center p-4"
    >
      <div className="w-screen flex justify-center items-center md:items-start gap-5 flex-col md:flex-row">
        <div className="flex flex-col gap-3 w-[280px]">
          <div className="shadow-lg">
            <Canvas ref={canvasRef} width={280} height={280} />
          </div>
          <button
            onClick={handleClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded self-stretch"
          >
            Predict
          </button>
          <button
            onClick={clear}
            className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded self-stretch"
          >
            Clear
          </button>
        </div>
        <div className="h-[280px] w-[280px] flex justify-center items-center text-9xl shadow-md">
          {prediction ?? "?"}
        </div>
      </div>
    </main>
  );
}
