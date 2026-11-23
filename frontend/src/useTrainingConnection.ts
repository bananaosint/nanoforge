import { useEffect } from 'react'
import { api, websocketUrl } from './api'
import type { Status } from './types'

export function useTrainingConnection(transport:'poll'|'websocket'|undefined,onStatus:(s:Status)=>void,onConnected:(value:boolean)=>void){
 useEffect(()=>{
  if(!transport)return
  let disposed=false,polling=false,lastMessage=0
  let socket:WebSocket|undefined,retry:ReturnType<typeof setTimeout>|undefined
  const poll=async()=>{
   if(disposed||polling||(transport==='websocket'&&Date.now()-lastMessage<4000))return
   polling=true
   try{const status=await api<Status>('/status');if(!disposed){onStatus(status);onConnected(true)}}
   catch{if(!disposed)onConnected(false)}finally{polling=false}
  }
  const connect=()=>{
   if(disposed)return
   socket=new WebSocket(websocketUrl())
   socket.onmessage=event=>{if(!disposed){lastMessage=Date.now();onStatus(JSON.parse(event.data) as Status);onConnected(true)}}
   socket.onclose=()=>{if(!disposed){void poll();retry=setTimeout(connect,2000)}}
  }
  if(transport==='websocket')connect()
  void poll()
  const timer=setInterval(()=>void poll(),1000)
  return()=>{disposed=true;clearInterval(timer);if(retry)clearTimeout(retry);socket?.close()}
 },[transport,onStatus,onConnected])
}
