function getRating() {
  element = document.getElementById("review-input")
  review = element.value
  element.value=""
  input = {
    'review': review
  }
  console.log('get rating ', review)
  fetch('http://127.0.0.1:5000/ratings/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(input),
  })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      if (data.status == 200) {
        console.log('Success:', data);
        renderRating(data);
      }
    })
    .catch((error) => {
      console.error('Error:', error);
    });
}

function renderRating(_data) {
  table = document.getElementById("review-table")
  var row = table.insertRow(1);
  var cell1 = row.insertCell(0);
  var cell2 = row.insertCell(1);
  var cell3 = row.insertCell(2);
  var cell4 = row.insertCell(3);
  var cell5 = row.insertCell(4);
  var cell6 = row.insertCell(5);
  var cell7 = row.insertCell(6);
  cell1.innerHTML = _data.review;
  cell2.innerHTML = _data.rating;
  cell3.innerHTML = _data.pr1;
  cell4.innerHTML = _data.pr2;
  cell5.innerHTML = _data.pr3;
  cell6.innerHTML = _data.pr4;
  cell7.innerHTML = _data.pr5;
}
