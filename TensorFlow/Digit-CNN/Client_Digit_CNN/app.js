new Vue({
  el: '#app',
  data: {
    mouse: {
      current: {
        x: 0,
        y: 0
      },
      previous: {
        x: 0,
        y: 0
      },
      down: false
    },
    loading: false,
    result: null,
    request: null,
    response: null,
  },
  computed: {
    currentMouse: function () {
      var rect = this.$refs['canvas'].getBoundingClientRect();

      return {
        x: this.mouse.current.x - rect.left,
        y: this.mouse.current.y - rect.top
      }
    }
  },
  methods: {
    send: function () {
      const vm = this
      const payload = {
        instances: [{
          b64: this.$refs['canvas'].toDataURL('image/png').replace('data:image/png;base64,', '')
        }]
      };

      this.request = JSON.stringify(payload, null, 2)
      this.loading = true

      axios.post('/api', payload)
        .then(function (res) {
          vm.result = res.data.predictions[0].classes
          vm.response = JSON.stringify(res.data, null, 2)
        })
        .catch(function (err) {
          console.log(err)
          vm.result = 'Error'
        })
        .finally(function () {
          vm.loading = false
        });
    },
    draw: function (event) {
      if (this.mouse.down) {
        var ctx = this.$refs['canvas'].getContext("2d");
        ctx.lineTo(this.currentMouse.x, this.currentMouse.y);
        ctx.strokeStyle = "#464646";
        ctx.lineWidth = 2;
        ctx.stroke()
      }

    },
    handleMouseDown: function (event) {
      this.mouse.down = true;
      this.mouse.current = {
        x: event.pageX,
        y: event.pageY
      }

      var ctx = this.$refs['canvas'].getContext("2d");
      ctx.moveTo(this.currentMouse.x, this.currentMouse.y)
    },
    handleMouseUp: function () {
      this.mouse.down = false;
    },
    handleMouseMove: function (event) {
      this.mouse.current = {
        x: event.pageX,
        y: event.pageY
      }

      this.draw(event)
    }
  },
  ready: function () {
    var ctx = this.$refs['canvas'].getContext("2d");
    ctx.translate(0.5, 0.5);
    ctx.imageSmoothingEnabled = false;
    // this.draw();
  }
});
