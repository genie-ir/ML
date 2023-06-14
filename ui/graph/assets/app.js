var lasthtml_contact_wrapper = ''
var counter = 0
var data_params = {}
var instance_params = {}
var cy = undefined
var last_G = undefined
var nodes = {}
var edges = {}
var selected = []

$(document). bind("contextmenu",function(e){ return false; })
$(document).ready(() => {
    $('#cy').show()
    $('#cy2').hide()
    get_data()
    $('button.menu__item').mousedown(function(event) {
        btn_name = $(this).attr('name')  
        // console.log(btn_name)
        if (event.which == 1) {
          $('body').css('background-color', $(this).css('--bgColorItem'))
          if (btn_name != undefined) {
              $('#cy').show()
              $('#cy2').hide()
              last_G = btn_name
              if (nodes[last_G] == undefined) {
                nodes[last_G] = []
              }
              if (edges[last_G] == undefined) {
                edges[last_G] = []
              }
              make_graph()
          } else {
            $('#cy').hide()
            $('#cy2').show()
          }
        } else {
          if ($(this).hasClass('active') && btn_name != undefined) {
              $('#cy').show()
              $('#cy2').hide()
              last_G = btn_name
              make_node((nodes[last_G] || []).length)
          }
        }
    })
    $('.btn_edit').on('click', function(event) { // TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      lasthtml_contact_wrapper = $('div.contact-wrapper').html()
      $('div.contact-wrapper').html('')
    })
})

function get_data() {
  $.ajax({
    url: 'https://raw.githubusercontent.com/genie-ir/ML/main/dependency/snippets.json',
    type: 'GET',
    dataType: 'json',
    success: function(res) {
      res.forEach(element => {
        if (element.content.startsWith('-')) {
          e_name = element.name
          e_content = element.content.slice(1).replaceAll(' ', '').replaceAll('\t', '').split('\n')
          e_cat = e_content[0].split(':')[0].split('.')[0]
          if (e_cat == '') {
            e_cat = 'passive/active'
          } else if (e_cat == 'input') {
            e_cat = 'IO'
          }
          // console.log(e_cat, e_name, e_content)
          if (data_params[e_cat] == undefined) {
            data_params[e_cat] = []
          }
          data_params[e_cat].push({
            e_cat: e_cat,
            e_name: e_name,
            e_content: e_content
          })
        }
      });
    }
  });
}
function make_node(name) {
    if (last_G == undefined) {
        return undefined
    }
    if (Array.isArray(nodes[last_G]) == false) {
        nodes[last_G] = []
    }
    L = nodes[last_G].length
    if (L > 0) {
      position = {
        x: nodes[last_G][L-1].position.x + .5 * 100,
        y: nodes[last_G][L-1].position.y
      }
      // console.log(position)
    } else {
      position = {x: 0, y: 0}
    }
    newid = ++counter
    node_value = { 
        data: { id: newid, name: '' },
        position: position
    }
    instance_params[newid] = { // default value
      'e_cat': 'IO',
      'e_name': 'input',
      'e_content': ["input:[z,randn,[3,256,256]]#inputz",
            "startdim:1",
            "enddim:mmd"
          ]
    }
    nodes[last_G].push(node_value)
    make_graph()
    return node_value
}
function mke_edge(source, target) {
    e =  { data: { id: ++counter, source: source, target: target } }
    if (Array.isArray(edges[last_G]) == false) {
      edges[last_G] = []
    }
    edges[last_G].push(e)
    make_graph()
    return e
}
function make_graph() {
    cy = cytoscape({
        container: document.getElementById('cy'),
      
        boxSelectionEnabled: false,
      
        style: [
          {
            selector: 'node',
            css: {
              'content': 'data(name)',
              'text-valign': 'center',
              'text-halign': 'center',
              // 'font-size': '.009em',
              // 'width': '.01em',
              // 'height': '.01em',
              // 'background-color': 'brown'
            }
          },
          {
            selector: ":selected",
            css: {
              // "background-color": "gold",
              "line-color": "black",
              "target-arrow-color": "black",
              "source-arrow-color": "black"
            }
          },
          {
            selector: ':parent',
            css: {
              'text-valign': 'top',
              'text-halign': 'center',
            }
          },
          {
            selector: 'edge',
            css: {
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              // 'font-size': '.009em',
              // 'width': '.01em',
              // 'height': '.01em',
            }
          }
        ],
      
        elements: {
          nodes: nodes[last_G],
          edges: edges[last_G]
        },
      
        layout: {
          name: 'preset',
          padding: 5
        }
    })
    cy.on('cxttap', "node", function(e) { 
      nodeid = this.id()
      $('#ne_name').html(instance_params[nodeid].e_cat + ' | ' + instance_params[nodeid].e_name)
      // console.log(instance_params[nodeid].e_content)
      tagshtml = ''
      for (let index = 1; index < instance_params[nodeid].e_content.length; index++) {
        let item_splited = instance_params[nodeid].e_content[index].split(':')
        console.log(item_splited)
        tagshtml += '<div class="form-item"><input id="'+item_splited[0]+'" type="text" placeholder="'+item_splited[0]+'"/><label class="lbl-floating" for="'+item_splited[0]+'">'+item_splited[0]+'</label><i class="fab2 icon text-white fas fa-times-circle"></i></div>'
      }
      $('div#ne_param').html(tagshtml)
      

      $('#formbtn').click()
    });
    cy.on('mousedown', 'node', function (e) {
      nodeid = this.id()
      // console.log('mousedown')
      if (selected.length == 0) {
        selected.push(nodeid)
      } else {
        if (window.event.ctrlKey) {
          s = selected[0]
          selected = []
          mke_edge(s, nodeid)
        } else {
          selected = [nodeid]
        }
      }
      // console.log(selected)
    })
    
    return cy
}