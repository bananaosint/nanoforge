export async function api<T>(path:string, method='GET', body?:unknown):Promise<T>{
  const response=await fetch('/api'+path,{method,headers:body instanceof FormData?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)})
  if(!response.ok){const data=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail))}
  return response.json()
}
