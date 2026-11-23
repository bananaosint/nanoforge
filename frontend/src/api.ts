export const apiUrl=(path:string)=>new URL('api'+path,document.baseURI).toString()
export const websocketUrl=()=>{const url=new URL('ws/training',document.baseURI);url.protocol=location.protocol==='https:'?'wss:':'ws:';return url.toString()}
export async function api<T>(path:string, method='GET', body?:unknown):Promise<T>{
  const response=await fetch(apiUrl(path),{method,headers:body instanceof FormData?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)})
  if(!response.ok){const data=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail))}
  return response.json()
}
