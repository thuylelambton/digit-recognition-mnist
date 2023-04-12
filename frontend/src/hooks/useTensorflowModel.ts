import { useQuery } from "react-query";
import * as tf from "@tensorflow/tfjs";

export const useTensorflowModel = (path: string) => {
  const { data: model, ...query } = useQuery(
    ["mnist-model", path],
    ({ queryKey }) => {
      return tf.loadLayersModel(queryKey[1]);
    },
    {
      staleTime: Infinity,
    }
  );
  return { model, ...query };
};
