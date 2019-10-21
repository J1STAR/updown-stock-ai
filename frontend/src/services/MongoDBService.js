import axios from 'axios';

export default {
    getUsers: function () {
        return axios.get('api/users/')
            .then(response => response.data.data)
            .catch(error => console.log(error));
    },
    postUser: function (name, age) {
        return axios.post('api/users/', {name: name, age: age})
            .then(response => response.data.data)
            .catch(error => console.log(error))
    }
}