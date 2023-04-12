const loadedScripts: Record<string, boolean> = {};

export const init = (url: string, id: string) => {
  return new Promise<void>((resolve, reject) => {
    let lib = window.document.getElementById(id) as HTMLScriptElement | null;
    if (lib) {
      // The lib is already initialized and the script tag is appended to DOM.
      // Will not init again and try call resolve accordingly the state of previous initialization.
      if (loadedScripts[id] === true) {
        // The script is loaded. Safe to resolve immediately.
        resolve();
        return;
      } else {
        // The script is loading. Append resolve() to previous onload function
        // then they will excute together when the script get loaded.
        const prevOnload = lib.onload;
        lib.onload = (e) => {
          if (typeof prevOnload === "function") {
            // @ts-expect-error
            prevOnload(lib, e);
          }
          resolve();
        };
      }
    } else {
      lib = window.document.createElement("script");
      lib.async = true;
      lib.onload = () => {
        loadedScripts[id] = true;
        resolve();
      };
      lib.onerror = reject;
      lib.id = id;
      lib.src = url;
      window.document.head.appendChild(lib);
    }
  });
};
