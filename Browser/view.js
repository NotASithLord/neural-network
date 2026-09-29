(() => {
  class Orbit {
    constructor(){this.reset();}
    reset(){this.yaw=-.35;this.pitch=.18;this.zoom=1;}
    project(x,y,z,cx,cy,scale=1){const a=x*Math.cos(this.yaw)+z*Math.sin(this.yaw),b=z*Math.cos(this.yaw)-x*Math.sin(this.yaw),v=y*Math.cos(this.pitch)-b*Math.sin(this.pitch),depth=y*Math.sin(this.pitch)+b*Math.cos(this.pitch),p=700/(700+depth*scale);return{x:cx+a*scale*p*this.zoom,y:cy+v*scale*p*this.zoom,z:depth,p};}
  }

 globalThis.NeuralView={Orbit};
})();
