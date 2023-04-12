import { init } from "@/utils/lib";
import { useQuery } from "react-query";

export const useFabricJs = () => {
  return useQuery(
    "fabric-js",
    () =>
      init(
        "https://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.4.0/fabric.min.js",
        "fabric.js"
      ),
    { staleTime: Infinity }
  );
};
