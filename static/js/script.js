const draggables = document.querySelectorAll('.draggable');
const container = document.querySelectorAll('.container');


draggables.forEach(draggable => {
  draggable.addEventListener('dragstart', ()=> {
    draggable.classList.add('dragging');
  });
});

draggables.forEach(draggable => {
  draggable.addEventListener('dragend', ()=> {
    draggable.classList.remove('dragging');
  });
});

container.forEach(container => {
  container.addEventListener('dragover', e => {
    e.preventDefault();
    const draggable = document.querySelector('.dragging');
    container.appendChild(draggable);   
  });
});
