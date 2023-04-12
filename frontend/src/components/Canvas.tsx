import { useFabricJs } from "@/hooks/useFabricJs";
import { init } from "@/utils/lib";
import {
  forwardRef,
  HTMLProps,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { useQuery } from "react-query";

export interface CanvasRef {
  getCanvas: () => fabric.Canvas | undefined;
}

export interface CanvasProps extends HTMLProps<HTMLCanvasElement> {}

export const Canvas = forwardRef<CanvasRef, CanvasProps>(function InnerCanvas(
  props,
  _ref
) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const { isLoading } = useFabricJs();
  const [canvas, setCanvas] = useState<fabric.Canvas>();

  useEffect(() => {
    if (isLoading) return;
    const canvas = new window.fabric.Canvas(ref.current);
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.width = 14;
    canvas.freeDrawingBrush.color = "#fff";
    canvas.backgroundColor = "#000";
    const ctx = canvas.getContext();
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width!, canvas.height!);
    setCanvas((c) => c ?? canvas);
  }, [isLoading]);

  useImperativeHandle(_ref, () => ({
    getCanvas: () => canvas,
  }));

  return <canvas ref={ref} {...props} />;
});
